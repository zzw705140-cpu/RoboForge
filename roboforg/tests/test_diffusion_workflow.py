from dataclasses import replace

import jax
import numpy as np
import pytest

from roboforg.agent.diffusion_policy import DataNormalizer
from roboforg.workflows.diffusion_policy.common import (
    DiffusionWorkflowConfig,
    checkpoint_setup,
    create_policy,
    initialize_state,
    restore_ema_params,
    restore_training_state,
    save_checkpoint,
)


def _small_setup():
    config = replace(
        DiffusionWorkflowConfig(),
        batch_size=1,
        diffusion_step_embedding_dim=8,
        down_channels=(8, 16, 32),
        n_groups=4,
    )
    normalizer = DataNormalizer()
    policy = create_policy(config, normalizer=normalizer)
    observations = {
        "state": np.zeros((1, 2, 23), dtype=np.float32),
        "front": np.zeros((1, 2, 32, 32, 3), dtype=np.uint8),
        "wrist": np.zeros((1, 2, 32, 32, 3), dtype=np.uint8),
    }
    state = initialize_state(
        policy, config, jax.random.key(0), observations, pretrained_path=None,
    )
    return config, normalizer, policy, observations, state


def test_ema_only_checkpoint_supports_evaluation_but_not_resume(tmp_path):
    config, normalizer, policy, observations, state = _small_setup()
    path = tmp_path / "epoch_0001.ckpt"
    save_checkpoint(
        path, state=state, epoch=0, config=config, normalizer=normalizer,
        metrics={"loss": 1.0}, save_training_state=False,
    )

    loaded_config, _, metadata = checkpoint_setup(path)
    assert loaded_config == config
    assert metadata["has_training_state"] is False
    initialized = policy.init_parameters(jax.random.key(1), observations)
    restored = restore_ema_params(path, initialized)
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(state.ema_params)
    with pytest.raises(ValueError, match="EMA weights only"):
        restore_training_state(path, state)


def test_full_checkpoint_restores_training_step(tmp_path):
    config, normalizer, _, _, state = _small_setup()
    state = state.replace(step=state.step + 7)
    path = tmp_path / "epoch_0001.ckpt"
    save_checkpoint(
        path, state=state, epoch=0, config=config, normalizer=normalizer,
        metrics={}, save_training_state=True,
    )
    restored = restore_training_state(path, state.replace(step=state.step - 7))
    assert int(restored.step) == 7
