from typing import Any, Callable, Sequence, Optional

import jax
import flax
import flax.linen as nn
import optax                    # 优化器，默认为 Adam 优化器

from roboforg.networks.common import Params, PRNGKey

##############################################################################
#                                 Model                                      #
##############################################################################
#负责创建网络，保存参数，加载参数，执行向前传播等通用模型操作


@flax.struct.dataclass
class Model:
    """模型类，封装模型参数、优化器状态、"""

    step: int                                                                           # 训练步数
    apply_fn: Callable = flax.struct.field(pytree_node=False)                           # 前向计算函数
    params: Params                                                                      # actor, critic 等网络参数
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)   # 优化器
    opt_state: Optional[optax.OptState] = None                                          # 优化器状态

    @classmethod
    def create(
            cls,
            *,
            network_def: nn.Module,
            inputs: Sequence[Any],                                                      # 输入数据, 用于初始化网络参数
            rng: PRNGKey,
            tx: Optional[optax.GradientTransformation] = None,
        ) -> 'Model':
        """创建模型实例, 并初始化网络参数和优化器状态"""

        # 初始化网络参数
        params = network_def.init(rng, *inputs)["params"]

        # 判断是否需要创建优化器状态: target_critic 不需要优化器
        opt_state = tx.init(params) if tx is not None else None

        # 返回模型实例
        return cls(
            step=0,
            apply_fn=network_def.apply,
            params=params,
            tx=tx,
            opt_state=opt_state
        )

    def __call__(self, *args, **kwds):
        return self.apply_fn({'params': self.params}, *args, **kwds)

    def apply(self, params, *args, **kwds):
        """调用模型的前向计算函数"""
        return self.apply_fn({'params': params}, *args, **kwds)

    def apply_gradient(self, grads):
        """基于 grads 更新 model 参数，优化器状态"""

        if self.tx is None or self.opt_state is None:
            raise ValueError("Cannot update a TrainState without an optimizer.")

        # 使用优化器更新 "参数" 和 "优化器状态"
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state)

    def apply_loss_fn(self, loss_fn, *, pmap_axis=None):
        """基于 loss_fn 更新 model 参数"""

        # 计算损失函数和梯度
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.params)

        # pmap_axis is not None, 说明是分布式训练, 需要对梯度进行平均
        if pmap_axis is not None:
            loss  = jax.lax.pmean(loss,  axis_name=pmap_axis)
            grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            aux   = jax.lax.pmean(aux,   axis_name=pmap_axis)

        # 应用梯度更新参数和优化器状态
        return self.apply_gradient(grads), {"loss": loss, **aux}
