"""连接观测编码、噪声预测、训练损失与 DDIM 动作采样。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from roboforg.agent.diffusion_policy.data_normalizer import DataNormalizer
from roboforg.agent.diffusion_policy.noise_scheduler import DiffusionScheduler
from roboforg.networks.diffusion_policy import DiffusionUNet, ObservationEncoder


# ============================================================================
# 完整网络：观测只编码一次，反向扩散循环只重复调用 U-Net
# ============================================================================

"""
DiffusionPolicyNetwork
  → 接收已经归一化的观测和带噪动作
  → ObservationEncoder 编码图像与机器人状态
  → 得到全局观测条件
  → DiffusionUNet 根据观测条件和扩散步预测噪声
  → 输出预测噪声 [B,16,7]
"""

class DiffusionPolicyNetwork(nn.Module):
    """组合观测编码器和条件 U-Net，并提供可独立调用的两个阶段。"""

    camera_keys: tuple[str, ...] = ("front", "wrist")
    state_dim: int = 23
    n_obs_steps: int = 2
    action_dim: int = 7
    diffusion_step_embedding_dim: int = 128
    down_channels: Sequence[int] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    predict_scale: bool = True

    # 创建一套共享参数的观测编码器和 U-Net，供不同方法分别调用。
    def setup(self):
        # 接收已归一化的观测和带噪动作
        observation_dim = len(self.camera_keys) * 512 + self.state_dim
        global_condition_dim = self.n_obs_steps * observation_dim

        # 编码机器人图像与机器人状态
        self.observation_encoder = ObservationEncoder(camera_keys=self.camera_keys, 
                                                      state_dim=self.state_dim,
                                                      n_obs_steps=self.n_obs_steps)
        self.unet = DiffusionUNet(action_dim=self.action_dim, 
                                  global_condition_dim=global_condition_dim,
                                  diffusion_step_embedding_dim=self.diffusion_step_embedding_dim,
                                  down_channels=self.down_channels, 
                                  kernel_size=self.kernel_size,
                                  n_groups=self.n_groups, 
                                  predict_scale=self.predict_scale)

    # 将多帧图像和状态编码一次，并按时间顺序展开为全局条件。
    def encode_observation(self, observations: Mapping[str, jax.Array]) -> jax.Array:
        """编码一次观测"""

        # 将两帧双相机图像和机器人状态编码为 [B, 2, 1047] 的特征
        features = self.observation_encoder(observations)
        # 将两帧特征展开为 [B, 2094] 的全局观测条件
        return features.reshape(features.shape[0], -1)

    # 复用已经得到的全局观测条件，预测当前扩散步中的动作噪声。
    def denoise(self, noisy_actions: jax.Array, timestep: jax.Array | int, global_condition: jax.Array) -> jax.Array:
        """定义单次噪声预测方法"""
        # 接收带噪动作，当前扩散不和观测条件，输出预测噪声

        # 预测当前带噪动作中包含的噪声，输出形状仍为 [B,16,7]
        return self.unet(noisy_actions, timestep, global_condition)

    # 训练时依次编码一次观测并执行一次噪声预测。
    def __call__(self, observations: Mapping[str, jax.Array], 
                 noisy_actions: jax.Array,
                 timestep: jax.Array | int) -> jax.Array:
        
        global_condition = self.encode_observation(observations)
        return self.denoise(noisy_actions, timestep, global_condition)


# ============================================================================
# 算法接口：归一化、训练目标、DDIM采样和动作截取
# ============================================================================

"""
DiffusionPolicy
→ 训练时：
    → 归一化数据
    → 给演示动作加噪
    → 调用 DiffusionPolicyNetwork 预测噪声
    → 计算预测噪声与真实噪声的 MSE

→ 推理时：
    → 编码一次观测
    → 从随机动作噪声开始
    → 循环调用 U-Net 和 DDIM Scheduler 去噪8次
    → 得到16步动作
    → 反归一化
    → 截取8步执行
"""

class DiffusionPolicy:
    """组织网络和 Scheduler；参数更新、EMA 与 checkpoint 由 workflow 管理。"""

    # 保存固定配置，并创建无参数的算法辅助对象。
    def __init__(self, *, camera_keys: tuple[str, ...] = ("front", "wrist"), 
                 state_dim:                     int = 23,
                 n_obs_steps:                   int = 2, 
                 horizon:                       int = 16,
                 n_action_steps:                int = 8, 
                 action_dim:                    int = 7,
                 num_inference_steps:           int = 8, 
                 diffusion_step_embedding_dim:  int = 128,
                 down_channels:                 Sequence[int] = (256, 512, 1024), 
                 kernel_size:                   int = 5,
                 n_groups:                      int = 8, 
                 predict_scale:                  bool = True, 
                 normalizer:                    DataNormalizer | None = None,
                 scheduler:                     DiffusionScheduler | None = None
                 ) -> None:
        
        if n_obs_steps <= 0 or horizon <= 0 or n_action_steps <= 0 or action_dim <= 0:
            raise ValueError("Observation, prediction, execution and action dimensions must be positive.")
        if n_obs_steps - 1 + n_action_steps > horizon:
            raise ValueError("n_obs_steps - 1 + n_action_steps must be <= horizon.")

        self.camera_keys = tuple(camera_keys)
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.normalizer = DataNormalizer() if normalizer is None else normalizer
        self.scheduler = DiffusionScheduler() if scheduler is None else scheduler
        self.scheduler.get_inference_timesteps(self.num_inference_steps)
        if self.normalizer.action_scale.shape[0] != self.action_dim:
            raise ValueError("Normalizer action dimension must match action_dim.")

        # 创建去噪网络
        self.network = DiffusionPolicyNetwork(camera_keys=self.camera_keys, state_dim=state_dim,
                                              n_obs_steps=n_obs_steps, action_dim=action_dim,
                                              diffusion_step_embedding_dim=diffusion_step_embedding_dim,
                                              down_channels=tuple(down_channels), kernel_size=kernel_size,
                                              n_groups=n_groups, predict_scale=predict_scale)

    # 在算法入口统一归一化图像和状态，保证训练与推理采用同一套统计量。
    def normalize_observations(self, observations: Mapping[str, jax.Array]) -> dict[str, jax.Array]:

        # 取出状态观测并归一化，先放入 normalized 字典
        normalized = {"state": self.normalizer.normalize_state(observations["state"])}

        # 遍历对应相机图像并归一化，然后放进字典
        for key in self.camera_keys:
            normalized[key] = self.normalizer.normalize_image(observations[key], image_key=key)
        return normalized

    # 使用真实接口形状初始化视觉编码器和 U-Net 的完整参数树。
    def init_parameters(self, 
                        rng: jax.Array, 
                        observations: Mapping[str, jax.Array]
                        ):
        
        # 先归一化输入观测
        normalized_observations = self.normalize_observations(observations)
        batch_size = normalized_observations["state"].shape[0]
        # 创建全零的动作序列，创建一批扩散步数样例
        noisy_actions = jnp.zeros((batch_size, self.horizon, self.action_dim), dtype=jnp.float32)
        timesteps = jnp.zeros((batch_size,), dtype=jnp.int32)

        return self.network.init(rng, normalized_observations, noisy_actions, timesteps)["params"]


    # 编码一次观测，再用固定条件完成全部确定性 DDIM 去噪步骤。
    # 定义动作序列生成的函数
    def sample_action_sequence(self, 
                               params, observations: Mapping[str, jax.Array], 
                               rng: jax.Array,
                               num_inference_steps: int | None = None           #指定去噪次数
                               ) -> jax.Array:
        
        inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        # 获取一次执行去噪所需的扩散步数序列
        timesteps = self.scheduler.get_inference_timesteps(inference_steps)
        normalized_observations = self.normalize_observations(observations)
        # 执行一次观测编码，供后续多次使用
        global_condition = self.network.apply({"params": params}, normalized_observations,method=self.network.encode_observation)
        batch_size = global_condition.shape[0]
        # 生成形状为 [批次大小, 预测长度, 动作维度] 的标准高斯噪声，作为去噪起点
        initial_sample = jax.random.normal(rng, (batch_size, self.horizon, self.action_dim), dtype=jnp.float32)

        # scan 只反复调用 U-Net；ResNet 得到的 global_condition 在循环外计算并复用。
        def denoise_step(sample, timestep):
            predicted_noise = self.network.apply({"params": params}, sample, timestep, global_condition,
                                                 method=self.network.denoise)
            # 让DDIM调度器根据预测噪声，将当前动作序列更新为下一去噪阶段的结果
            previous_sample = self.scheduler.ddim_step(predicted_noise, timestep, sample, inference_steps)
            return previous_sample, None

        # 按照 timesteps 依次执行去噪，每轮接收上一轮的结果，最终得到完整动作序列。
        final_sample, _ = jax.lax.scan(denoise_step, initial_sample, timesteps)
        # 返回生成的动作序列
        return final_sample

    #################
    #    训练入口    #
    #################

    # 随机选择逐样本扩散步，构造噪声目标并返回可反向传播的 MSE 损失。
    def compute_loss(self, 
                     params, observations: Mapping[str, jax.Array],   #当前网络参数，状态
                     actions: jax.Array,                              #真实动作
                     rng: jax.Array,                                  #用于随机选择扩散步数和生成噪声的随机数键
                     return_full: bool = False                        #不额外返回完整的噪声数组和扩散步数
                     ):

        # 归一化动作，图像，状态
        normalized_observations = self.normalize_observations(observations)
        normalized_actions = self.normalizer.normalize_action(actions)

        # 检查动作序列的预测长度和动作维度是否与配置一致
        if normalized_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("actions must have shape [B, horizon, action_dim].")
        
        """
        将随机数键拆成两个，分别用于抽取扩散步数和生成噪声。
        读取当前批次中动作序列的样本数量。
        为每条动作序列随机选择一个扩散步数，范围为 0 到训练扩散总步数减一。
        生成与动作数组形状相同的标准高斯噪声，作为网络的预测目标。
        根据选定的扩散步数，将真实动作与噪声按对应比例混合，得到带噪动作。
        使用当前网络参数，根据观测、带噪动作和扩散步数预测噪声。
        计算预测噪声与目标噪声之间的均方误差，作为训练损失。
        """

        timestep_rng, noise_rng = jax.random.split(rng)
        batch_size              = normalized_actions.shape[0]
        timesteps               = jax.random.randint(timestep_rng, (batch_size,), 0, self.scheduler.num_train_timesteps, dtype=jnp.int32)
        target_noise            = jax.random.normal(noise_rng, normalized_actions.shape, dtype=normalized_actions.dtype)
        noisy_actions           = self.scheduler.add_noise(normalized_actions, target_noise, timesteps)
        predicted_noise         = self.network.apply({"params": params}, normalized_observations, noisy_actions, timesteps)
        loss                    = jnp.mean((predicted_noise - target_noise) ** 2)

        # 记录损失、预测噪声均值、目标噪声均值和扩散步数均值，供训练日志使用
        metrics = {"loss": loss, "predicted_noise_mean": jnp.mean(predicted_noise),
                   "target_noise_mean": jnp.mean(target_noise), "timestep_mean": jnp.mean(timesteps)}
        
        if return_full:
            metrics.update({"predicted_noise": predicted_noise, "target_noise": target_noise, "timesteps": timesteps})
        return loss, metrics

    #################
    #    推理入口    #
    #################

    # 生成完整动作序列，反归一化后截取当前需要执行的连续动作。
    def predict_action(self, 
                       params, observations: Mapping[str, jax.Array], 
                       rng: jax.Array,
                       num_inference_steps: int | None = None
                       ) -> dict[str, jax.Array]:
        # 生成并反归一化动作

        # 通过多次去噪生成归一化空间中的完整动作序列
        normalized_sequence = self.sample_action_sequence(params, observations, rng, num_inference_steps)
        # 将动作反归一化
        action_prediction = self.normalizer.denormalize_action(normalized_sequence)
        # 将最新观测对应的位置作为执行起点，跳过序列中与历史观测对应的动作位置
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {"action": action_prediction[:, start:end], "action_pred": action_prediction}
