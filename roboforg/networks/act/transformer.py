"""通用 Transformer 主体：ACT 默认 Post-Norm，输入统一为 [B,L,D]。

位置编码由外部传入，在每层 Q/K 投影之前相加，V 不加位置编码。
mask 采用布尔值 True=禁止关注（与 PyTorch 一致）；不支持浮点加性 mask。
训练时 apply(..., train=True, rngs={"dropout": key})；推理默认关闭 Dropout。
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


# 检查序列形状，返回 batch 大小；token 不能是空序列。
def _check_sequence(x, token_dim, name):
    if x.ndim != 3 or min(x.shape) <= 0 or x.shape[-1] != token_dim:
        raise ValueError(f"{name} must have non-empty shape [B,L,{token_dim}].")
    return x.shape[0]


# 检查位置编码与内容一一对应；允许全 batch 共用一份位置编码。
def _with_position(x, pos):
    if pos is None:
        return x
    if pos.shape not in ((1, x.shape[1], x.shape[2]), x.shape):
        raise ValueError("pos must have shape [1,L,D] or [B,L,D] matching tokens.")
    return x + pos


# 共有网络设计：前馈网络与残差归一化，由编码器和解码器继承。
class TransformerBlockBase(nn.Module):
    token_dim: int = 256
    num_heads: int = 8
    feedforward_dim: int = 2048
    num_layers: int = 4
    dropout_rate: float = 0.1

    # 构建模块时检查网络宽度、层数、头数和 Dropout 范围。
    def setup(self):
        if self.token_dim <= 0 or self.num_heads <= 0 or self.token_dim % self.num_heads:
            raise ValueError("token_dim must be positive and divisible by num_heads > 0.")
        if self.feedforward_dim <= 0 or self.num_layers <= 0:
            raise ValueError("feedforward_dim and num_layers must be positive.")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0,1).")

    # 每个 token 独立经过 D→F→D；name 区分不同层，各层使用独立参数。
    def feed_forward(self, x, *, train, name):
        x = nn.Dense(self.feedforward_dim, kernel_init=nn.initializers.xavier_uniform(), name=f"{name}_in")(x)
        x = nn.relu(x)
        # 顺序与 ACT 一致：第一层全连接、ReLU、Dropout、第二层全连接。
        x = nn.Dropout(self.dropout_rate, name=f"{name}_dropout")(x, deterministic=not train)
        return nn.Dense(self.token_dim, kernel_init=nn.initializers.xavier_uniform(), name=f"{name}_out")(x)

    # 将分支更新加回原输入，再对每个 token 的 D 个特征做 LayerNorm。
    def residual_norm(self, x, update, *, train, name):
        update = nn.Dropout(self.dropout_rate, name=f"{name}_residual_dropout")(update, deterministic=not train)
        # epsilon 对齐 PyTorch LayerNorm 默认值；各分支的缩放与偏置独立。
        return nn.LayerNorm(epsilon=1e-5, name=f"{name}_norm")(x + update)


# 多头注意力：自注意力和交叉注意力共用代码，各次实例化分别持有参数。
class MultiHeadAttention(nn.Module):
    token_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    # 输入是待投影的特征；输出 [B,Lq,D]，长度由 query 决定。
    @nn.compact
    def __call__(self, query, key, value, *, mask=None, key_padding_mask=None, train=False):
        if self.token_dim <= 0 or self.num_heads <= 0 or self.token_dim % self.num_heads:
            raise ValueError("token_dim must be positive and divisible by num_heads > 0.")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0,1).")
        batch = _check_sequence(query, self.token_dim, "query")
        _check_sequence(key, self.token_dim, "key")
        _check_sequence(value, self.token_dim, "value")
        if key.shape != value.shape or key.shape[0] != batch:
            raise ValueError("key/value must have matching shapes and the query batch size.")
        lq, lk = query.shape[1], key.shape[1]
        head_dim = self.token_dim // self.num_heads

        # 三套独立全连接权重生成 Q/K/V，再从 [B,L,D] 变为 [B,heads,L,head_dim]。
        q = nn.Dense(self.token_dim, kernel_init=nn.initializers.xavier_uniform(), name="query")(query).reshape(batch, lq, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = nn.Dense(self.token_dim, kernel_init=nn.initializers.xavier_uniform(), name="key")(key).reshape(batch, lk, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = nn.Dense(self.token_dim, kernel_init=nn.initializers.xavier_uniform(), name="value")(value).reshape(batch, lk, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # 每个头计算所有查询对所有键的打分：[B,heads,Lq,Lk]。
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(float(head_dim))
        blocked = jnp.zeros((batch, self.num_heads, lq, lk), dtype=jnp.bool_)
        if mask is not None:
            # 支持共享 [Lq,Lk]，或可广播的 [B或1,heads或1,Lq,Lk]。
            if mask.dtype != jnp.bool_:
                raise ValueError("mask must be boolean: True means blocked.")
            if mask.shape == (lq, lk):
                mask = mask[None, None, :, :]
            elif mask.ndim != 4 or mask.shape[0] not in (1, batch) or mask.shape[1] not in (1, self.num_heads) or mask.shape[2:] != (lq, lk):
                raise ValueError("mask must be [Lq,Lk] or [B or 1,heads or 1,Lq,Lk].")
            blocked = blocked | mask
        if key_padding_mask is not None:
            # padding 只屏蔽作为信息来源的键；不自动清零对应查询位置的最终输出。
            if key_padding_mask.dtype != jnp.bool_ or key_padding_mask.shape != (batch, lk):
                raise ValueError("key_padding_mask must be boolean [B,Lk].")
            blocked = blocked | key_padding_mask[:, None, None, :]

        # 屏蔽后沿键维归一化；全屏蔽行的权重置零，避免 NaN。
        scores = jnp.where(blocked, jnp.finfo(scores.dtype).min, scores)
        weights = jax.nn.softmax(scores, axis=-1)
        weights = jnp.where(blocked, 0.0, weights)
        weights = nn.Dropout(self.dropout_rate, name="attention_dropout")(weights, deterministic=not train)

        # 用权重对 V 求和，合并各头后做输出投影；全屏蔽行可能保留输出层偏置。
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, lq, self.token_dim)
        return nn.Dense(self.token_dim, kernel_init=nn.initializers.xavier_uniform(), name="output")(attended)


# 编码器：同一个类实现单层流程及多层堆叠，可分别实例化给 CVAE 和观测编码器。
class TransformerEncoder(TransformerBlockBase):

    # 一层：带位置的自注意力，然后前馈；两处分支各有残差与归一化。
    def _layer(self, x, pos, *, mask, key_padding_mask, train, index):
        name = f"layer_{index}"
        qk = _with_position(x, pos)
        update = MultiHeadAttention(self.token_dim, self.num_heads, self.dropout_rate, name=f"{name}_attention")(
            qk, qk, x, mask=mask, key_padding_mask=key_padding_mask, train=train)
        x = self.residual_norm(x, update, train=train, name=f"{name}_attention")
        update = self.feed_forward(x, train=train, name=f"{name}_ffn")
        return self.residual_norm(x, update, train=train, name=f"{name}_ffn")

    # 输入/输出 [B,L,D]；pos 单独传入，每层再次用于 Q/K 计算。
    @nn.compact
    def __call__(self, tokens, pos=None, *, mask=None, key_padding_mask=None, train=False):
        _check_sequence(tokens, self.token_dim, "tokens")
        x = tokens
        # 不同 index 生成不同参数名；多层是复用算法步骤，不是共享权重。
        for index in range(self.num_layers):
            x = self._layer(x, pos, mask=mask, key_padding_mask=key_padding_mask, train=train, index=index)
        return x


# 解码器：在自注意力和前馈之间增加交叉注意力，从编码器 memory 中提取信息。
class TransformerDecoder(TransformerBlockBase):

    # 一层：查询之间自注意力 → 查询读取 memory → 前馈加工。
    def _layer(self, x, memory, pos, query_pos, *, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, train, index):
        name = f"layer_{index}"
        qk = _with_position(x, query_pos)
        update = MultiHeadAttention(self.token_dim, self.num_heads, self.dropout_rate, name=f"{name}_self_attention")(
            qk, qk, x, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, train=train)
        x = self.residual_norm(x, update, train=train, name=f"{name}_self_attention")

        # Q 来自解码特征，K/V 来自 memory；位置只加到 Q/K 的投影输入上。
        update = MultiHeadAttention(self.token_dim, self.num_heads, self.dropout_rate, name=f"{name}_cross_attention")(
            _with_position(x, query_pos), _with_position(memory, pos), memory,
            mask=memory_mask, key_padding_mask=memory_key_padding_mask, train=train)
        x = self.residual_norm(x, update, train=train, name=f"{name}_cross_attention")
        update = self.feed_forward(x, train=train, name=f"{name}_ffn")
        return self.residual_norm(x, update, train=train, name=f"{name}_ffn")

    # tgt [B,k,D]、memory [B,L,D]；返回 [B,k,D]，动作投影留给 ACT 上层。
    @nn.compact
    def __call__(self, tgt, memory, pos=None, query_pos=None, *, tgt_mask=None, memory_mask=None,
                 tgt_key_padding_mask=None, memory_key_padding_mask=None, train=False):
        batch = _check_sequence(tgt, self.token_dim, "tgt")
        if _check_sequence(memory, self.token_dim, "memory") != batch:
            raise ValueError("tgt and memory must have the same batch size.")
        x = tgt
        # ACT 初始 tgt 为零，query_pos 为可学习动作查询；默认不添加因果 mask。
        for index in range(self.num_layers):
            x = self._layer(x, memory, pos, query_pos, tgt_mask=tgt_mask, memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                            train=train, index=index)
        # ACT 解码器堆叠结束后还有一次额外 LayerNorm，仅返回最后一层结果。
        return nn.LayerNorm(epsilon=1e-5, name="final_norm")(x)
