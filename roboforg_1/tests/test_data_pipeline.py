"""基础数据层测试：通用工具、观测兼容、存储校验和采集文件往返。

使用小型模拟数据验证边界，不启动仿真窗口或人工输入设备。
"""

import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from roboforg_1.data.chunk_bc_replay_buffer import ChunkBCReplayBuffer
from roboforg_1.data.data_store import InMemoryDataStore
from roboforg_1.data.observation import canonicalize_observation, canonicalize_trajectory, load_dataset
from roboforg_1.data.utils import stack_tree, tree_asarray, tree_leading_length, tree_take


@pytest.fixture
def trajectory() -> dict:
    """包含两台相机及三步动作的规范轨迹。"""
    return {
        "observations": {
            "state": np.arange(8, dtype=np.float32).reshape(4, 2),
            "images": {
                "front": np.arange(48, dtype=np.uint8).reshape(4, 2, 2, 3),
                "wrist": np.zeros((4, 2, 2, 3), dtype=np.uint8),
            },
        },
        "actions": np.arange(6, dtype=np.float32).reshape(3, 2),
        "rewards": np.array([0, 0, 1], dtype=np.float32),
        "masks": np.array([1, 1, 0], dtype=np.float32),
        "dones": np.array([False, False, True]),
    }


# --------------------------- 通用数组树工具 --------------------------- #

def test_tree_operations_keep_nested_fields_and_dtypes(trajectory) -> None:
    tree = trajectory["observations"]
    assert tree_leading_length(tree) == 4
    assert tree_asarray(tree)["state"] is tree["state"]
    converted = tree_asarray({"nested": {"value": [[1, 2], [3, 4]]}})
    np.testing.assert_array_equal(converted["nested"]["value"], [[1, 2], [3, 4]])
    selected = tree_take(tree, [3, 0, 0])
    np.testing.assert_array_equal(selected["state"], tree["state"][[3, 0, 0]])
    np.testing.assert_array_equal(selected["images"]["front"], tree["images"]["front"][[3, 0, 0]])
    # 字典插入顺序不同不应影响结构相同的样本堆叠。
    first = tree_take(tree, 0)
    second = {"images": tree_take(tree, 1)["images"], "state": tree["state"][1]}
    stacked = stack_tree([first, second])
    np.testing.assert_array_equal(stacked["state"], tree["state"][:2])
    assert stacked["images"]["front"].dtype == np.uint8


@pytest.mark.parametrize("tree", [{}, {"x": 1}, {"x": [1, 2], "y": [1]}])
def test_invalid_tree_lengths_are_rejected(tree) -> None:
    with pytest.raises(ValueError):
        tree_leading_length(tree)


def test_incompatible_stacks_and_unsupported_leaves_are_rejected() -> None:
    for values in ([], [{"x": np.zeros(2)}, {"y": np.zeros(2)}], [np.zeros(2), np.zeros(3)]):
        with pytest.raises(ValueError):
            stack_tree(values)
    with pytest.raises(TypeError):
        tree_take({"x": [1, 2]}, 0)


# --------------------------- 新旧观测格式与文件读取 --------------------------- #

def test_canonicalization_preserves_values_and_is_idempotent(trajectory) -> None:
    observations = trajectory["observations"]
    legacy = {"state": observations["state"], **observations["images"], "extra": {"value": np.ones((4, 1))}}
    result = canonicalize_observation(legacy)
    assert set(result) == {"state", "images", "extra"}
    assert result["images"]["front"] is legacy["front"]
    assert "images" not in legacy
    repeated = canonicalize_observation(result)
    assert repeated["state"] is result["state"]
    assert repeated["images"]["wrist"] is result["images"]["wrist"]
    assert canonicalize_trajectory({**trajectory, "observations": legacy})["actions"] is trajectory["actions"]
    single = canonicalize_observation(tree_take(legacy, 0))
    assert single["state"].shape == (2,)
    assert single["images"]["front"].shape == (2, 2, 3)


@pytest.mark.parametrize("observation,error", [
    ([], TypeError),
    ({"front": np.zeros(1)}, KeyError),
    ({"state": np.zeros(1)}, ValueError),
    ({"state": np.zeros(1), "images": []}, TypeError),
    ({"state": np.zeros(1), "front": np.zeros(1), "images": {"front": np.zeros(1)}}, ValueError),
])
def test_invalid_observations_are_rejected(observation, error) -> None:
    with pytest.raises(error):
        canonicalize_observation(observation)


def test_custom_camera_names_and_malformed_files(tmp_path) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert canonicalize_observation({"state": np.zeros(2), "side": image}, camera_keys=("side",))["images"]["side"] is image
    path = tmp_path / "invalid.pkl"
    for payload in ([], {}, {"trajectories": [{"actions": np.zeros((1, 2))}]}):
        path.write_bytes(pickle.dumps(payload))
        with pytest.raises((ValueError, KeyError)):
            load_dataset(path)


# --------------------------- 存储容量、完整性与结构一致性 --------------------------- #

def test_store_import_capacity_and_reference_semantics(trajectory) -> None:
    store = InMemoryDataStore.from_trajectories(iter([trajectory]), capacity=6)
    assert (len(store), store.num_trajectories, store.capacity) == (3, 1, 6)
    assert store.get_trajectory(-1)["actions"] is trajectory["actions"]
    store.add_trajectory(copy.deepcopy(trajectory))
    np.testing.assert_array_equal(store.trajectory_lengths, [3, 3])
    with pytest.raises(ValueError, match="capacity"):
        store.add_trajectory(trajectory)
    assert (len(store), store.num_trajectories) == (6, 2)
    with pytest.raises(ValueError, match="capacity"):
        InMemoryDataStore.from_trajectories([trajectory], capacity=2)
    with pytest.raises(ValueError):
        InMemoryDataStore.from_trajectories([])
    for index in (2, -3):
        with pytest.raises(IndexError):
            store.get_trajectory(index)
    for index in (True, 0.5):
        with pytest.raises(TypeError):
            store.get_trajectory(index)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_is_rejected(capacity) -> None:
    with pytest.raises(ValueError):
        InMemoryDataStore(capacity)


@pytest.mark.parametrize("problem", ["missing", "observation_count", "camera_count", "actions", "rewards", "masks", "early_done", "no_done"])
def test_invalid_trajectory_is_rejected_without_mutation(trajectory, problem) -> None:
    store = InMemoryDataStore.from_trajectories([trajectory], capacity=10)
    invalid = copy.deepcopy(trajectory)
    if problem == "missing":
        del invalid["actions"]
    elif problem == "observation_count":
        invalid["observations"]["state"] = invalid["observations"]["state"][:-1]
    elif problem == "camera_count":
        invalid["observations"]["images"]["front"] = invalid["observations"]["images"]["front"][:-1]
    elif problem == "actions":
        invalid["actions"] = np.zeros(3)
    elif problem == "rewards":
        invalid["rewards"] = invalid["rewards"][:, None]
    elif problem == "masks":
        invalid["masks"][1] = 0.5
    elif problem == "early_done":
        invalid["dones"][0] = True
    else:
        invalid["dones"][-1] = False
    with pytest.raises((ValueError, KeyError)):
        store.add_trajectory(invalid)
    assert (len(store), store.num_trajectories) == (3, 1)


@pytest.mark.parametrize("problem", ["action_dim", "state_dim", "camera_keys", "image_size", "image_dtype"])
def test_incompatible_trajectories_are_rejected_at_import(trajectory, problem) -> None:
    invalid = copy.deepcopy(trajectory)
    if problem == "action_dim":
        invalid["actions"] = np.zeros((3, 7), dtype=np.float32)
    elif problem == "state_dim":
        invalid["observations"]["state"] = np.zeros((4, 23), dtype=np.float32)
    elif problem == "camera_keys":
        del invalid["observations"]["images"]["wrist"]
    elif problem == "image_size":
        invalid["observations"]["images"]["front"] = np.zeros((4, 3, 3, 3), dtype=np.uint8)
    else:
        invalid["observations"]["images"]["front"] = invalid["observations"]["images"]["front"].astype(np.float32)
    store = InMemoryDataStore.from_trajectories([trajectory], capacity=10)
    with pytest.raises(ValueError, match="schema"):
        store.add_trajectory(invalid)
    assert (len(store), store.num_trajectories) == (3, 1)
    with pytest.raises(ValueError, match="schema"):
        InMemoryDataStore.from_trajectories([trajectory, invalid])


def test_time_limit_done_allows_bootstrap(trajectory) -> None:
    trajectory["masks"][-1] = 1
    store = InMemoryDataStore.from_trajectories([trajectory])
    assert store.get_trajectory(0)["dones"][-1]
    assert store.get_trajectory(0)["masks"][-1] == 1


# --------------------------- 采集保存 → 读取 → Store → chunk --------------------------- #

def test_collector_file_round_trip(tmp_path, trajectory) -> None:
    from roboforg_1.data.collect_data.collect_data import _copy_tree, _finalize_trajectory, _save_dataset

    observations = trajectory["observations"]
    legacy = {"state": observations["state"], **observations["images"]}
    frames = [_copy_tree(tree_take(legacy, step)) for step in range(4)]
    # 采集缓冲区快照与环境持有的数组互不影响。
    frames[0]["state"][0] = -10
    assert observations["state"][0, 0] == 0
    frames[0]["state"][0] = 0
    collected = _finalize_trajectory(frames, list(trajectory["actions"]), list(trajectory["rewards"]), list(trajectory["masks"]), list(trajectory["dones"]))
    path = _save_dataset(trajectories=[collected], statistics={"successful_episodes": 1}, split="train", output_root=tmp_path, output_dir_name="round_trip")
    payload = load_dataset(path)
    assert payload["split"] == "train"
    assert payload["statistics"] == {"successful_episodes": 1}
    store = InMemoryDataStore.from_trajectories(payload["trajectories"])
    batch = next(ChunkBCReplayBuffer(store).get_epoch_iterator(batch_size=3, observation_horizon=2, action_horizon=4, shuffle=False))
    np.testing.assert_array_equal(batch["actions"][:, 0], trajectory["actions"])
    np.testing.assert_array_equal(batch["action_valid"].sum(axis=1), [3, 2, 1])
    np.testing.assert_array_equal(store.get_trajectory(0)["observations"]["images"]["front"], observations["images"]["front"])


def test_legacy_file_uses_the_same_pipeline(tmp_path, trajectory) -> None:
    observations = trajectory["observations"]
    legacy = {**trajectory, "observations": {"state": observations["state"], **observations["images"]}}
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps({"trajectories": [legacy], "split": "eval"}))
    payload = load_dataset(path)
    store = InMemoryDataStore.from_trajectories(payload["trajectories"])
    batch = ChunkBCReplayBuffer(store, seed=0).sample_chunk(2, observation_horizon=1, action_horizon=2)
    assert set(batch["observations"]["images"]) == {"front", "wrist"}
    assert batch["actions"].shape == (2, 2, 2)


@pytest.mark.parametrize("truncated", [False, True])
def test_collection_records_executed_actions_and_filters_episodes(monkeypatch, truncated) -> None:
    from roboforg_1.data.collect_data import collect_data as collector

    class ScriptedEnv:
        """前三轮分别失败、作废、未开始；第四轮成功，仅有效动作后开始录制。"""

        action_space = SimpleNamespace(shape=(7,), dtype=np.float32)

        def __init__(self):
            self.unwrapped = self
            self.config = SimpleNamespace(hz=1e9, gripper_threshold=0.5)
            self.episode = -1

        def observation(self):
            value = self.episode * 10 + self.step_index
            return {"state": np.array([value], dtype=np.float32), "front": np.full((2, 2, 3), value, dtype=np.uint8)}

        def reset(self, seed=None):
            self.episode += 1
            self.step_index = 0
            return self.observation(), {}

        def step(self, hold_action):
            self.step_index += 1
            done = self.episode < 3 or self.step_index == 3
            moving = self.episode != 2 and (self.episode < 3 or self.step_index >= 2)
            executed = np.zeros(7, dtype=np.float32)
            executed[0] = self.step_index * 0.1 if moving else 0
            executed[-1] = -1
            info = {"is_takeover_active": True, "intervene_action": executed, "is_success": self.episode > 0, "rerecord_episode": self.episode == 1}
            timed_out = done and self.episode == 3 and truncated
            return self.observation(), float(done), done and not timed_out, timed_out, info

    monkeypatch.setattr(collector.time, "sleep", lambda seconds: None)
    trajectories, statistics, complete = collector.collect_demonstrations(ScriptedEnv(), successes_needed=1, seed=0)
    assert complete
    assert statistics == {"successful_episodes": 1, "attempted_episodes": 4, "total_steps": 6}
    trajectory = trajectories[0]
    np.testing.assert_array_equal(trajectory["observations"]["state"][:, 0], [31, 32, 33])
    np.testing.assert_allclose(trajectory["actions"][:, 0], [0.2, 0.3])
    np.testing.assert_array_equal(trajectory["dones"], [False, True])
    np.testing.assert_array_equal(trajectory["masks"], [1, 1 if truncated else 0])
    assert InMemoryDataStore.from_trajectories(trajectories).num_transitions == 2
