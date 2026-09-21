from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import requests
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict

from roboforg.model.flax_model import Model

###########################################################
#               加载残差图像编码器，主要负责视觉输入            #
###########################################################


_RESNET10_FILE_NAME = "resnet10_params.pkl"
_RESNET10_URL = "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT_PATH = _PROJECT_ROOT / ".resnet_params" / _RESNET10_FILE_NAME


def _download_checkpoint(url: str, destination: Path) -> None:
    """把预训练 ResNet 权重从网络下载到本地缓存"""

    # 创建目标目录
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 下载到临时 .tmp 文件
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")

    try:
        # 下载成功命名为正式文件
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        temporary_path.replace(destination)
    finally:
        # 失败则删除不完整的临时文件
        if temporary_path.exists():
            temporary_path.unlink()


def _find_subtree_paths(
    tree: Mapping[str, Any],
    subtree_name: str,
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    """Return every path whose final key equals ``subtree_name``."""

    paths = []
    for key, value in tree.items():
        path = (*prefix, key)
        if key == subtree_name:
            paths.append(path)
        if isinstance(value, Mapping):
            paths.extend(_find_subtree_paths(value, subtree_name, path))
    return paths


def _get_subtree(tree: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    subtree = tree
    for key in path:
        subtree = subtree[key]
    return subtree


def _set_subtree(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    subtree = tree
    for key in path[:-1]:
        subtree = subtree[key]
    subtree[path[-1]] = value


def _merge_backbone_params(initialized_params: Any, pretrained_params: Any) -> Any:
    """Replace every initialized backbone leaf with its pretrained value."""

    initialized_state = flax.serialization.to_state_dict(initialized_params)
    pretrained_state = flax.serialization.to_state_dict(pretrained_params)

    # Accept checkpoints serialized either as a bare parameter tree or as
    # {"params": parameter_tree}.
    if "params" in pretrained_state and "params" not in initialized_state:
        pretrained_state = pretrained_state["params"]

    initialized_flat = flatten_dict(initialized_state)
    pretrained_flat = flatten_dict(pretrained_state)

    missing = sorted(set(initialized_flat) - set(pretrained_flat))
    if missing:
        formatted = ["/".join(path) for path in missing]
        raise ValueError(
            "The pretrained ResNet checkpoint is missing parameters required "
            f"by the initialized backbone: {formatted}"
        )

    merged_flat = dict(initialized_flat)
    for path, initialized_value in initialized_flat.items():
        pretrained_value = pretrained_flat[path]
        if initialized_value.shape != pretrained_value.shape:
            raise ValueError(
                f"Shape mismatch for {'/'.join(path)}: initialized "
                f"{initialized_value.shape}, pretrained {pretrained_value.shape}."
            )
        merged_flat[path] = jnp.asarray(
            pretrained_value,
            dtype=initialized_value.dtype,
        )

    return unflatten_dict(merged_flat)


def load_resnet10_params(encoder_model: Model, checkpoint_path: str | Path | None = None, *, download_url: str = _RESNET10_URL) -> Model:
    """加载预训练 ResNet-10 参数并返回更新后的 Encoder Model"""

    # 用户未指定路径时，将参数缓存在项目根目录的 .resnet_params 文件夹中。
    if checkpoint_path is None:
        checkpoint_path = _DEFAULT_CHECKPOINT_PATH
    else:
        checkpoint_path = Path(checkpoint_path).expanduser()

    # 检查文件是否存在，不存在就调用，并从指定 URL 下载并保存；存在则复用，避免重复下载
    if not checkpoint_path.exists():
        print(f"Downloading pretrained ResNet-10 weights to {checkpoint_path}")
        _download_checkpoint(download_url, checkpoint_path)

    # 从本地读取预训练参数, ResNet 参数树加载到内存中的 pretrained_params
    with checkpoint_path.open("rb") as file:
        pretrained_params = pickle.load(file)

    # 将 Flax 参数转化为普通嵌套字典，方便搜索，修改
    params_state = flax.serialization.to_state_dict(encoder_model.params)

    # 递归查找参数树中所有名为 pretrained_encoder 的子树路径，未找到则说明 Encoder 中不存在预训练 ResNet，或参数名称不匹配，报错
    backbone_paths = _find_subtree_paths(params_state, "pretrained_encoder")
    if not backbone_paths:
        raise ValueError("No 'pretrained_encoder' subtree was found in encoder_model.params.")

    # 依次替换每个 ResNet backbone 参数子树
    for path in backbone_paths:
        initialized_backbone = _get_subtree(params_state, path)

        # 检查参数名称，形状，并使用预训练权重替换对应参数
        loaded_backbone = _merge_backbone_params(initialized_backbone, pretrained_params)

        # 将替换完成的 ResNet 参数放回完整的 encoder 参数树
        _set_subtree(params_state, path, loaded_backbone)

    # 修改后的普通参数字典恢复成与原 encoder_model.params 相同的 Flax 参数结构
    new_params = flax.serialization.from_state_dict(encoder_model.params, params_state)

    # 取出预训练参数树中的所有数组，并累加元素数量，得到预训练 ResNet 的总参数量
    parameter_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(pretrained_params))

    print(
        f"Loaded {parameter_count / 1e6:.2f}M pretrained ResNet-10 "
        f"parameters into {len(backbone_paths)} shared backbone subtree(s)."
    )

    return encoder_model.replace(params=new_params)


def load_resnet10_into_policy_state(
    policy_state: TrainState,
    checkpoint_path: str | Path | None = None,
    *,
    download_url: str = _RESNET10_URL,
) -> TrainState:
    """将预训练 ResNet-10 权重替换到 ACTPolicy 的 ``params['backbone']``。"""

    # 只能在首次优化前加载；训练中覆盖参数会使 AdamW 的历史动量与新权重不匹配。
    if int(policy_state.step) != 0:
        raise ValueError("Load pretrained ResNet-10 before the first policy optimizer update.")

    # 未指定路径时使用项目统一缓存；文件不存在才下载，已下载文件不会重复请求网络。
    if checkpoint_path is None:
        checkpoint_path = _DEFAULT_CHECKPOINT_PATH
    else:
        checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        print(f"Downloading pretrained ResNet-10 weights to {checkpoint_path}")
        _download_checkpoint(download_url, checkpoint_path)

    # 读取 SEPO/SERL 发布的预训练参数树；文件损坏或格式错误会在此直接报错。
    with checkpoint_path.open("rb") as file:
        pretrained_params = pickle.load(file)

    # ACTPolicy 将共享图像主干注册为顶层 backbone；其余 token、Transformer、动作头绝不修改。
    params_state = flax.serialization.to_state_dict(policy_state.params)
    if "backbone" not in params_state:
        raise KeyError(
            "ACT policy parameters have no 'backbone' subtree. "
            "Create ACTPolicy with its ResNet backbone before loading pretrained weights."
        )

    # 逐叶检查名称和形状后，使用预训练值替换随机初始化的 ResNet 参数。
    params_state["backbone"] = _merge_backbone_params(
        params_state["backbone"], pretrained_params
    )
    new_params = flax.serialization.from_state_dict(policy_state.params, params_state)

    # 加载发生在首次训练更新前；优化器状态形状未变，可安全保留其全零动量。
    parameter_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(pretrained_params))
    print(
        f"Loaded {parameter_count / 1e6:.2f}M pretrained ResNet-10 parameters "
        "into ACT policy backbone."
    )
    return policy_state.replace(params=new_params)
