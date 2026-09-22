"""训练调度与仿真评估回归测试，不启动训练或真实图形窗口。"""

import contextlib
import io
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from roboforg.workflows.act import run_act, train_act
from roboforg.workflows.act.checkpoint_paths import latest_checkpoint, resolve_run
from roboforg.workflows.act.training_logger import ACTTrainingLogger


class ACTWorkflowTest(unittest.TestCase):
    def test_rollout_modes_and_cleanup(self):
        agent = SimpleNamespace(policy=SimpleNamespace(action_horizon=4, action_dim=7))
        for viewer, count in ((False, 40), (True, 10)):
            env = Mock()
            env.unwrapped.config.max_episode_length = 150
            results = [{"success": i % 2 == 0, "steps": 150, "return": 0.0}
                       for i in range(count)]
            output = io.StringIO()
            with patch.object(run_act, "make_evaluation_environment", return_value=env), \
                 patch.object(run_act, "run_rollout", side_effect=results) as rollout, \
                 contextlib.redirect_stdout(output):
                metrics = run_act.evaluate_policy(agent, show_viewer=viewer, seed=17)
            self.assertEqual(rollout.call_count, count)
            self.assertEqual([call.kwargs["seed"] for call in rollout.call_args_list], list(range(17, 17 + count)))
            self.assertTrue(all(call.kwargs["max_steps"] == 150 for call in rollout.call_args_list))
            env.close.assert_called_once()
            if viewer:
                self.assertIsNone(metrics)
                self.assertNotIn("success_rate", output.getvalue())
            else:
                self.assertEqual(metrics["success_rate"], 0.5)
                self.assertEqual(metrics["num_rollouts"], 40)

        env = Mock()
        with patch.object(run_act, "make_evaluation_environment", return_value=env), \
             patch.object(run_act, "run_rollout", side_effect=RuntimeError("simulation failed")):
            with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                run_act.evaluate_policy(agent)
        env.close.assert_called_once()

    def test_checkpoint_selection_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "act_k4_b32_run1"
            directory.mkdir()
            legacy = directory / "latest.ckpt"
            legacy.touch()
            self.assertEqual(resolve_run(root, 1), legacy)
            def save(path, **kwargs):
                path.touch()
            with patch.object(train_act, "save_checkpoint", side_effect=save), \
                 contextlib.redirect_stdout(io.StringIO()):
                for epoch in range(199, 1200, 200):
                    train_act.save_training_checkpoint(directory, agent=None, epoch=epoch, config=None, metrics={})
            self.assertEqual(len(list(directory.glob("epoch_*.ckpt"))), 5)
            self.assertFalse((directory / "epoch_0200.ckpt").exists())
            self.assertTrue(legacy.exists())
            self.assertEqual(latest_checkpoint(directory).name, "epoch_1200.ckpt")
            (root / "act_k5_b64_run1").mkdir()
            with self.assertRaisesRegex(ValueError, "found 2"):
                resolve_run(root, 1)
            with self.assertRaisesRegex(ValueError, "found 0"):
                resolve_run(root, 2)

    def test_training_schedule(self):
        # 跑实际 main 调度，替换耗时的网络、数据和环境；验证末轮不重复。
        for start, epochs, expected in ((0, 1000, [500, 1000]), (0, 750, [500, 750]), (400, 600, [500, 1000])):
            with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                argv = ["train_act", "--num-epochs", str(epochs), "--checkpoint-dir", tmp]
                if start:
                    argv += ["--resume", str(Path(tmp) / "epoch_0400.ckpt")]
                stack.enter_context(patch("sys.argv", argv))
                stack.enter_context(patch.object(train_act.jax.random, "key", return_value=0))
                stack.enter_context(patch.object(train_act.jax.random, "split", return_value=(0, 1)))
                config = train_act.ACTWorkflowConfig()
                agent = SimpleNamespace(policy_state=SimpleNamespace(step=0))
                logger = Mock()
                current = {"epoch": start}
                stack.enter_context(patch.object(train_act, "read_checkpoint_setup", return_value=(config, None, {})))
                stack.enter_context(patch.object(train_act, "load_checkpoint_into_agent", return_value=(agent, {"epoch": start - 1})))
                evaluated = []
                saved = []
                def epoch_pass(**kwargs):
                    if kwargs["training"]:
                        current["epoch"] += 1
                        agent.policy_state.step += 3
                    return agent, {"loss": 1.0}, kwargs["rng"], 3
                def evaluate(*args, **kwargs):
                    self.assertFalse(kwargs["show_viewer"])
                    self.assertEqual(kwargs["num_rollouts"], 40)
                    evaluated.append(current["epoch"])
                    return {"success_rate": 0.5, "num_rollouts": 40}
                for name, result in (("build_config", config), ("load_split_trajectories", []),
                                     ("make_chunk_buffer", None), ("fit_train_normalizer", None),
                                     ("sample_act_batch", None), ("create_act_agent", agent)):
                    stack.enter_context(patch.object(train_act, name, return_value=result))
                stack.enter_context(patch.object(train_act, "run_epoch", side_effect=epoch_pass))
                stack.enter_context(patch.object(train_act, "save_training_checkpoint", side_effect=lambda *a, **kw: saved.append(kw["epoch"] + 1)))
                stack.enter_context(patch.object(run_act, "evaluate_policy", side_effect=evaluate))
                stack.enter_context(patch.object(train_act.ACTTrainingLogger, "create", return_value=logger))
                train_act.main()
                self.assertEqual(evaluated, expected)
                self.assertEqual(saved, sorted({e for e in range(start + 1, start + epochs + 1) if e % 200 == 0} | {start + epochs}))
                self.assertEqual([call.kwargs["epoch"] + 1 for call in logger.log_rollout.call_args_list], expected)
                logger.finish.assert_called_once()

    def test_wandb_payload(self):
        backend = Mock()
        logger = ACTTrainingLogger(backend)
        logger.log_rollout({"success_rate": 0.75, "num_rollouts": 40}, step=1234, epoch=499)
        backend.log.assert_called_once_with({
            "train_step": 1234,
            "rollout": {"success_rate": 0.75, "num_rollouts": 40, "epoch": 500},
        })

    def test_shell_arguments(self):
        script = Path(run_act.__file__).with_name("test_act.sh")
        # 替换 python 命令来观察 shell 转发的参数，不加载模型或写入项目目录。
        command = '''
python() {
  if [[ "$1" == "-m" ]]; then printf '%s\\n' "$@";
  else printf '%s\\n' '/tmp/example/epoch_1000.ckpt'; fi
}
export -f python
bash "$@"
'''
        result = subprocess.run(["bash", "-c", command, "test", str(script), "--run", "3", "--show-viewer", "--num-rollouts", "7"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--show-viewer", result.stdout)
        self.assertIn("--num-rollouts\n7", result.stdout)
        self.assertIn("--checkpoint\n/tmp/example/epoch_1000.ckpt", result.stdout)
        for args in ([], ["--run", "0"], ["--run", "1", "--num-rollouts"], ["--bad"]):
            result = subprocess.run(["bash", str(script), *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
