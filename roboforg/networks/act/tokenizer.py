"""ACT 的特征投影、token 化、固定位置编码与可学习标记。"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from collections.abc import Mapping

#####################################
#              网络                 #
#####################################


# 基础网络：对最后一维做单层线性投影，状态、动作和视觉特征均可复用。
class FeatureProjection(nn.Module):
    token_dim: int

    # 保留 batch、时间、空间等前导维度，只将最后一维映射到 token_dim。
    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive.")
        if features.ndim < 2 or features.shape[-1] == 0:
            raise ValueError("features must have a non-empty final feature dimension and a batch dimension.")

        # 一层 y = Wx + b；不额外添加隐藏层、激活函数或归一化。
        return nn.Dense(self.token_dim, name="linear")(features)


# 状态 token 化：一个时刻的全部状态数共同组成一个 token。
class StateTokenizer(nn.Module):
    token_dim: int

    # 输入 [B,S]，输出 [B,1,D]；S 不写死为 23，便于适配其他状态。
    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        if state.ndim != 2 or state.shape[0] == 0:
            raise ValueError("state must have shape [B,S] with B > 0.")

        # 将状态投影到 D 维，并显式增加长度为 1 的 token 维。
        features = FeatureProjection(self.token_dim, name="projection")(state)
        return features[:, None, :]


# 图像 token 化：复用外部视觉主干，将每个空间位置转换为一个 token。
class ImageTokenizer(nn.Module):
    """backbone 必须返回 [B,H',W',C]，并支持 encode/train 参数。

    现有 ResNetEncoder 可设置 pre_pooling=False、pooling_method="none"、
    bottleneck_dim=None，返回可训练的特征图；pre_pooling=True 则阻断主干梯度。
    初始化本模块不会自动加载预训练权重，权重加载由后续训练入口负责。
    """

    backbone: nn.Module
    token_dim: int

    # 输入单路 RGB 图像 [B,H,W,3]，输出 [B,H'*W',D]。
    @nn.compact
    def __call__(self, images: jax.Array, train: bool = False) -> jax.Array:
        if images.ndim != 4 or images.shape[-1] != 3 or min(images.shape[:3]) <= 0:
            raise ValueError("images must have non-empty shape [B,H,W,3].")

        # 输入使用现有数据管线的 0~255 RGB；缩放和归一化由 ResNet 内部完成。
        features = self.backbone(images, encode=True, train=train)
        if features.ndim != 4 or features.shape[0] != images.shape[0] or min(features.shape[1:]) <= 0:
            raise ValueError("backbone must return a non-empty feature map [B,H',W',C].")

        # 同一线性层处理所有位置的通道特征，作用等价于 1×1 卷积。
        features = FeatureProjection(self.token_dim, name="projection")(features)

        # 按行排列空间位置，保留每个位置的 D 个特征；不进行空间加权汇总。
        batch_size, height, width, _ = features.shape
        return features.reshape(batch_size, height * width, self.token_dim)


#######################################
#            实现token化               #
#######################################


# 动作 token 化：真实动作 chunk 中每一步对应一个 token，供训练期 CVAE 使用。
class ActionTokenizer(nn.Module):
    token_dim: int

    # 输入 [B,k,A]，输出 [B,k,D]；当前 A=7，但不限制未来动作维度。
    @nn.compact
    def __call__(self, actions: jax.Array) -> jax.Array:
        if actions.ndim != 3 or min(actions.shape) <= 0:
            raise ValueError("actions must have non-empty shape [B,k,A].")

        # 所有时间步共享动作投影权重；与状态、视觉投影使用不同的参数实例。
        return FeatureProjection(self.token_dim, name="projection")(actions)


#######################################
#              位置编码                #
#######################################
# 固定一维位置编码：为 [CLS]、状态和动作序列生成逐位置的 D 维标记。
class SinusoidalPositionEncoding1D(nn.Module):
    token_dim: int

    # 输入序列长度 L，输出 [1,L,D]；长度为形状信息，JIT 时应保持静态。
    def __call__(self, length: int) -> jax.Array:
        if length <= 0 or self.token_dim <= 0 or self.token_dim % 2:
            raise ValueError("length must be positive and token_dim must be positive and even.")

        # 位置从 0 开始；每对 sin/cos 使用同一个频率，不创建可训练参数。
        positions = jnp.arange(length, dtype=jnp.float32)[:, None]
        frequencies = 10000.0 ** (jnp.arange(0, self.token_dim, 2, dtype=jnp.float32) / self.token_dim)
        angles = positions / frequencies

        # 偶数通道放 sin，奇数通道放 cos；前导 1 便于广播到整个 batch。
        encoding = jnp.stack((jnp.sin(angles), jnp.cos(angles)), axis=-1)
        return encoding.reshape(1, length, self.token_dim)


# 固定二维位置编码：行、列各占 D/2 个通道，使用原版 ACT 的归一化坐标。
class SinusoidalPositionEncoding2D(nn.Module):
    token_dim: int

    # 输入特征图 H'、W'，输出 [1,H'*W',D]；这里不是原始图像的尺寸。
    def __call__(self, height: int, width: int) -> jax.Array:
        if height <= 0 or width <= 0 or self.token_dim <= 0 or self.token_dim % 4:
            raise ValueError("height/width must be positive and token_dim must be positive and divisible by 4.")

        # 与 ACT 一致，从 1 开始计数并缩放到约 2π；当前使用无 padding 的完整图像。
        rows = jnp.arange(1, height + 1, dtype=jnp.float32) / (height + 1e-6) * (2 * jnp.pi)
        cols = jnp.arange(1, width + 1, dtype=jnp.float32) / (width + 1e-6) * (2 * jnp.pi)
        axis_dim = self.token_dim // 2
        frequencies = 10000.0 ** (jnp.arange(0, axis_dim, 2, dtype=jnp.float32) / axis_dim)
        row_angles = rows[:, None] / frequencies
        col_angles = cols[:, None] / frequencies

        # 分别生成行列编码，再广播到每个空间位置，先行编码、后列编码。
        row_encoding = jnp.stack((jnp.sin(row_angles), jnp.cos(row_angles)), axis=-1).reshape(height, axis_dim)
        col_encoding = jnp.stack((jnp.sin(col_angles), jnp.cos(col_angles)), axis=-1).reshape(width, axis_dim)
        grid = jnp.concatenate((
            jnp.broadcast_to(row_encoding[:, None, :], (height, width, axis_dim)),
            jnp.broadcast_to(col_encoding[None, :, :], (height, width, axis_dim)),
        ), axis=-1)

        # 与 ImageTokenizer 一样按行展平；多相机的 token 和位置编码须按同一顺序拼接。
        return grid.reshape(1, height * width, self.token_dim)


# 可学习 CLS：自身就是内容 token，后续放在 CVAE 输入序列的第一个位置。
class CLSToken(nn.Module):
    token_dim: int

    # 输出 [B,1,D]；整个 batch 共享同一个参数，而非各自学习不同的 CLS。
    @nn.compact
    def __call__(self, batch_size: int) -> jax.Array:
        if batch_size <= 0 or self.token_dim <= 0:
            raise ValueError("batch_size and token_dim must be positive.")

        # 直接创建可训练向量，不经过 Dense；采用与 PyTorch Embedding 默认相同的标准正态初始化。
        token = self.param("embedding", nn.initializers.normal(stddev=1.0), (1, 1, self.token_dim))
        return jnp.broadcast_to(token, (batch_size, 1, self.token_dim))


# 可学习动作查询：每个未来动作输出位置各有一个独立向量，作为解码器 query_pos。
class ActionQueries(nn.Module):
    num_queries: int
    token_dim: int

    # 输出 [B,k,D]；k 对应 action_horizon，不需要输入真实动作。
    @nn.compact
    def __call__(self, batch_size: int) -> jax.Array:
        if batch_size <= 0 or self.num_queries <= 0 or self.token_dim <= 0:
            raise ValueError("batch_size, num_queries and token_dim must be positive.")

        # k 个位置分别学习参数，在 batch 间共享；不把查询向量作为初始 tgt 内容。
        queries = self.param("embedding", nn.initializers.normal(stddev=1.0), (1, self.num_queries, self.token_dim))
        # 后续解码器使用 query_pos=本输出、tgt=jnp.zeros_like(本输出)。
        return jnp.broadcast_to(queries, (batch_size, self.num_queries, self.token_dim))


#######################################
#                拼接                 #
#######################################


# 检查内容 token 的 batch、序列长度和特征宽度；所有拼接均沿 axis=1。
def _check_tokens(name, tokens, batch_size, token_dim, length=None):
    if tokens.ndim != 3 or tokens.shape[0] != batch_size or tokens.shape[2] != token_dim:
        raise ValueError(f"{name} must have shape [{batch_size},N,{token_dim}].")
    if tokens.shape[1] <= 0 or (length is not None and tokens.shape[1] != length):
        raise ValueError(f"{name} has an invalid token sequence length.")


# CVAE 输入组装：接收已有 CLS、状态、动作 token，生成对应的一维位置编码。
class CVAEInputAssembler(nn.Module):
    token_dim: int

    # 返回内容 [B,k+2,D] 和位置 [1,k+2,D]；此处不进行相加。
    @nn.compact
    def __call__(self, cls_token, state_token, action_tokens):
        if state_token.ndim != 3 or state_token.shape[0] <= 0:
            raise ValueError("state_token must have shape [B,1,D] with B > 0.")
        batch_size = state_token.shape[0]
        _check_tokens("cls_token", cls_token, batch_size, self.token_dim, length=1)
        _check_tokens("state_token", state_token, batch_size, self.token_dim, length=1)
        _check_tokens("action_tokens", action_tokens, batch_size, self.token_dim)

        # 保留动作时间顺序；CLS 固定放在索引 0，供后续读取汇总特征。
        tokens = jnp.concatenate((cls_token, state_token, action_tokens), axis=1)
        pos = SinusoidalPositionEncoding1D(self.token_dim, name="position")(tokens.shape[1])
        return tokens, pos


# 观测编码器输入组装：z、状态、各相机 token 与位置编码保持完全相同的排列。
class ObservationInputAssembler(nn.Module):
    """视觉 token 与位置编码由外部生成；按 camera_keys 排列，不依赖字典顺序。

    image_tokens[key]: [B,N_camera,D]
    image_positions[key]: [1,N_camera,D]，来自对应特征图尺寸的二维位置编码。
    输入 z 是原始潜变量 [B,Z]；状态已是 token [B,1,D]。
    """

    token_dim: int
    camera_keys: tuple[str, ...] = ("front", "wrist")

    # 返回 [B,2+ΣN_camera,D] 的内容和 [1,2+ΣN_camera,D] 的位置编码。
    @nn.compact
    def __call__(self, z, state_token, image_tokens: Mapping, image_positions: Mapping):
        if state_token.ndim != 3 or state_token.shape[0] <= 0:
            raise ValueError("state_token must have shape [B,1,D] with B > 0.")
        batch_size = state_token.shape[0]
        _check_tokens("state_token", state_token, batch_size, self.token_dim, length=1)
        if z.ndim != 2 or z.shape[0] != batch_size or z.shape[1] <= 0:
            raise ValueError("z must have shape [B,Z] and match the state batch size.")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be non-empty and contain no duplicates.")

        # 将原始 z 从 [B,Z] 投影为 [B,1,D]；零 z 也需经过这个带偏置的投影。
        latent_token = FeatureProjection(self.token_dim, name="latent_projection")(z)[:, None, :]

        # 直接创建两个可学习位置标记，依次对应 z 和状态，在整个 batch 间共享。
        special_positions = self.param(
            "special_positions", nn.initializers.normal(stddev=1.0), (1, 2, self.token_dim)
        )
        token_parts = [latent_token, state_token]
        position_parts = [special_positions]

        # 相机可以有不同数量的 token，但各自的位置编码长度必须匹配。
        for key in self.camera_keys:
            if key not in image_tokens or key not in image_positions:
                raise KeyError(f"Missing image tokens or position encoding for {key!r}.")
            visual = image_tokens[key]
            position = image_positions[key]
            _check_tokens(f"image_tokens[{key}]", visual, batch_size, self.token_dim)
            _check_tokens(f"image_positions[{key}]", position, 1, self.token_dim, length=visual.shape[1])
            token_parts.append(visual)
            position_parts.append(position)

        # 两条序列分别拼接，位置相加留给 Transformer 的 Q/K 计算。
        tokens = jnp.concatenate(token_parts, axis=1)
        pos = jnp.concatenate(position_parts, axis=1)
        return tokens, pos
