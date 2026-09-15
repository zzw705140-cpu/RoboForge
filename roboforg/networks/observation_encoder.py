from typing import Iterable, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp


##############################################################################
#               Image and Proprioception Encoder Network                     #
##############################################################################
#观测/图像编码器

class ObservationEncoder(nn.Module):
    """将多帧图像和 proprio observation 编码成统一特征向量

    支持两种输入格式:

    格式 1: use_temporal_history = true      代码是否支持使用时间历史窗口
    
    推理阶段 (单个历史窗口):
        {
            "front": [T, H, W, C],
            "wrist": [T, H, W, C],
            "state": [T, state_dim],
        }
    训练阶段（批量历史窗口）：
        {
            "front": [B, T, H, W, C],
            "wrist": [B, T, H, W, C],
            "state": [B, T, state_dim],
        }

    格式 2: use_temporal_history = false

    推理阶段 (单个历史窗口):
        {
            "front": [H, W, C],
            "wrist": [H, W, C],
            "state": [state_dim],
        }
    训练阶段（批量历史窗口）：
        {
            "front": [B, H, W, C],
            "wrist": [B, H, W, C],
            "state": [B, state_dim],
        }

    当 use_temporal_history = true, 此时每一帧图像 [H, W, C] 分别经过共享 ResNet, 再在特征层融合时间维
    
    当前时间融合方式为 flatten: 
        [B,T,image_dim] → [B,T*image_dim]
        [B,T,proprio_dim] → [B,T*proprio_dim]

    输出格式：

        推理阶段：[feature_dim]
        训练阶段：[B, feature_dim]
    """

    image_encoders: Mapping[str, nn.Module]         # 每个相机对应一个图像 Encoder
    use_proprio: bool                               # 是否编码并拼接机器人本体状态

    proprio_latent_dim: int = 64                    # 每一帧 proprio 投影后的维度
    use_temporal_history: bool = True               # 代码是否支持使用时间历史窗口
    image_keys: Iterable[str] = ("image",)          # 需要编码的相机字段

    @nn.compact
    def __call__(self, observations: Mapping[str, jax.Array], train: bool = False) -> jax.Array:
        """编码 observation, 并在特征层融合相机、时间和 proprio 信息"""

        # 用第一个相机确定输入是单个历史窗口还是批量历史窗口；后续所有相机和 proprio 必须使用相同的 B、T 结构。
        input_is_batched = None
        batch_size = None
        observation_horizon = None
        encoded_images = []

        # -------------------------------------------------------------- #
        #                         图像特征编码                             #
        # -------------------------------------------------------------- #

        for image_key in self.image_keys:
            if image_key not in observations:
                raise KeyError(f"Observation has no image key {image_key!r}.")

            image = observations[image_key]

            if self.use_temporal_history:
                if image.ndim == 4:
                    # 推理：[T,H,W,C]。将 T 暂时作为 ResNet 的 batch 维
                    current_is_batched = False
                    current_batch_size = None
                    current_observation_horizon = image.shape[0]
                    image_batch = image                                     # [T,H,W,C]
                elif image.ndim == 5:
                    # 训练：[B,T,H,W,C] → [B*T,H,W,C]
                    current_is_batched = True
                    current_batch_size = image.shape[0]
                    current_observation_horizon = image.shape[1]
                    image_batch = image.reshape((-1, *image.shape[-3:]))    # [B*T, H, W, C]
                else:
                    raise ValueError(f"Expected temporal image {image_key!r} with shape [T,H,W,C] or [B,T,H,W,C], got {image.shape}.")
            else:
                # 兼容不使用历史窗口的调用；内部仍视作 T=1
                current_observation_horizon = 1
                if image.ndim == 3:
                    current_is_batched = False
                    current_batch_size = None
                    image_batch = image[None, ...]                          # [1,H,W,C]
                elif image.ndim == 4:
                    current_is_batched = True
                    current_batch_size = image.shape[0]
                    image_batch = image                                     # [B,H,W,C]
                else:
                    raise ValueError(f"Expected image {image_key!r} with shape [H,W,C] or [B,H,W,C], got {image.shape}.")

            # 确定是否是 batch, 并确定 B 和 T, 并检查后续其他 image 的 B T 是否和第一个 image 一致
            if input_is_batched is None:
                input_is_batched = current_is_batched
                batch_size = current_batch_size
                observation_horizon = current_observation_horizon
            elif (input_is_batched != current_is_batched or batch_size != current_batch_size or observation_horizon != current_observation_horizon):
                raise ValueError("All image observations must use the same batch and temporal dimensions.")

            # 每个相机的逐帧计算流程：
            #
            # [B*T,H,W,3]
            # → frozen ResNet backbone      
            # → SpatialLearnedEmbeddings
            # → Dense(256)
            # → LayerNorm + tanh
            # → [B*T,256]
            image_features = self.image_encoders[image_key](
                image_batch,
                encode=True,
                train=train,
            )

            if image_features.ndim != 2:
                raise ValueError(f"Image encoder {image_key!r} must return [N,D], got {image_features.shape}.")

            # 恢复 B、T 语义，并将时间特征按顺序展平。flatten 保留时间顺序
            if current_is_batched:
                image_features = image_features.reshape((current_batch_size, current_observation_horizon, -1))
                image_features = image_features.reshape((current_batch_size, -1))   # [B,T*D]
            else:
                image_features = image_features.reshape(-1)                         # [T*D]

            encoded_images.append(image_features)

        if not encoded_images:
            raise ValueError("EncoderNet requires at least one image key.")

        # 多相机特征沿最后一维拼接：
        #
        # 推理：K × [T*D]   → [K*T*D]
        # 训练：K × [B,T*D] → [B,K*T*D]
        features = jnp.concatenate(encoded_images, axis=-1)

        # -------------------------------------------------------------- #
        #                       Proprio 特征编码                           #
        # -------------------------------------------------------------- #

        if self.use_proprio:
            if "state" not in observations:
                raise KeyError("use_proprio=True, but observations has no 'state' key.")

            state = observations["state"]

            if self.use_temporal_history:
                if state.ndim == 2:
                    # 推理：[T,state_dim]，T 作为 Dense 的 batch 维
                    state_is_batched = False
                    state_batch_size = None
                    state_observation_horizon = state.shape[0]
                    state_batch = state                                 # [T,state_dim]
                elif state.ndim == 3:
                    # 训练：[B,T,state_dim] → [B*T,state_dim]
                    state_is_batched = True
                    state_batch_size = state.shape[0]
                    state_observation_horizon = state.shape[1]
                    state_batch = state.reshape((-1, state.shape[-1]))
                else:
                    raise ValueError(f"Expected temporal proprio state with shape [T,D] or [B,T,D], got {state.shape}.")
            else:
                state_observation_horizon = 1
                if state.ndim == 1:
                    state_is_batched = False
                    state_batch_size = None
                    state_batch = state[None, ...]                      # [1,state_dim]
                elif state.ndim == 2:
                    state_is_batched = True
                    state_batch_size = state.shape[0]
                    state_batch = state                                 # [B,state_dim]
                else:
                    raise ValueError(f"Expected proprio state with shape [D] or [B,D], got {state.shape}.")

            if (state_is_batched != input_is_batched or state_batch_size != batch_size or state_observation_horizon != observation_horizon):
                raise ValueError("Proprio state and image observations must use the same batch and temporal dimensions.")

            # 每个时间帧共享同一个 proprio 投影层：[B*T,state_dim] → Dense(64) → LayerNorm → tanh → [B*T,64]
            state_features = nn.Dense(
                features=self.proprio_latent_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                name="proprio_projection",
            )(state_batch)
            state_features = nn.LayerNorm(name="proprio_layer_norm")(state_features)
            state_features = nn.tanh(state_features)

            if state_is_batched:
                state_features = state_features.reshape((state_batch_size, state_observation_horizon, -1))
                state_features = state_features.reshape((state_batch_size, -1))     # [B,T*64]
            else:
                state_features = state_features.reshape(-1)                         # [T*64]

            features = jnp.concatenate([features, state_features], axis=-1)

        return features
