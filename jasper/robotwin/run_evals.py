"""
Evaluate a policy server on RoboTwin test episodes.

Reads test episodes (from test.txt split files) for each dataset config,
sends observation frames to a websocket policy server in chunks, executes
the returned actions in simulation, and reports per-config and overall
success rates.

Usage:
    python jasper/robotwin/run_evals.py --dataset-dir /home/ubuntu/workspace/RoboTwin/dataset

    python jasper/robotwin/run_evals.py --dataset-dir /home/ubuntu/workspace/RoboTwin/dataset \
                        --task-name adjust_bottle \
                        --server ws://localhost:8765
"""

import asyncio
import sys
import importlib
import math
import time
import numpy as np
import h5py
import yaml
import os
import imageio
import msgpack
import websockets
import cv2
from pathlib import Path
from collections import OrderedDict

orig_root = Path(__file__).resolve()
sim_root = "/home/ubuntu/workspace/RoboTwin"
if sim_root not in sys.path:
    sys.path.insert(0, sim_root)
os.chdir(sim_root)

from envs import CONFIGS_PATH

CHUNK_SIZE = 30


def decode_rgb(buf):
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


def load_env(task_name):
    mod = importlib.import_module(f"envs.{task_name}")
    return getattr(mod, task_name)()


def load_config(task_config):
    with open(f"./task_config/{task_config}.yml", "r") as f:
        args = yaml.safe_load(f)

    embodiment_type = args["embodiment"]
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        emb_types = yaml.safe_load(f)

    robot_file = emb_types[embodiment_type[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True

    with open(os.path.join(robot_file, "config.yml"), "r") as f:
        emb_config = yaml.safe_load(f)

    args["left_embodiment_config"] = emb_config
    args["right_embodiment_config"] = emb_config
    return args


def collect_test_episodes(dataset_dir):
    """Read test episodes grouped by subdirectory."""
    grouped = OrderedDict()
    for subdir in sorted(dataset_dir.iterdir()):
        if not subdir.is_dir():
            continue
        test_file = subdir / "test.txt"
        seed_file = subdir / "seed.txt"
        if not test_file.exists() or not seed_file.exists():
            continue

        seeds = list(map(int, seed_file.read_text().strip().split()))
        task_config = "demo_randomized" if "randomized" in subdir.name else "demo_clean"

        episodes = []
        with open(test_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep_path = subdir / line
                ep_idx = int(ep_path.stem.replace("episode", ""))
                episodes.append((str(ep_path), seeds[ep_idx], task_config))

        if episodes:
            grouped[subdir.name] = episodes

    return grouped


def serialize_frame(img):
    return {
        "data": img.tobytes(),
        "shape": list(img.shape),
        "dtype": str(img.dtype),
    }


async def request_actions(ws, frames):
    msg = msgpack.packb({"frames": [serialize_frame(f) for f in frames]})
    await ws.send(msg)
    response = msgpack.unpackb(await ws.recv(), raw=False)

    if "error" in response:
        raise RuntimeError(f"Server error: {response['error']}")

    act = response["actions"]
    return np.frombuffer(act["data"], dtype=act["dtype"]).reshape(act["shape"])


async def evaluate_episode(
    ws,
    action_mean,
    action_std,
    ep_path,
    seed,
    task_config,
    task_name,
    save_dir,
):
    env = load_env(task_name)
    args = load_config(task_config)
    args["task_name"] = task_name
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["save_data"] = False
    env.setup_demo(now_ep_num=0, seed=seed, **args)

    # Load all observation frames from HDF5
    with h5py.File(ep_path, "r") as f:
        rgb_data = f["observation/head_camera/rgb"]
        T = len(rgb_data)
        all_frames = [decode_rgb(rgb_data[i]) for i in range(T)]

    num_chunks = math.ceil(T / CHUNK_SIZE)

    cam_frames = {}
    success = False
    step_count = 0
    t_start = time.time()
    idx = 0

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, T)
        chunk = all_frames[start:end]

        # Pad last chunk by repeating the final frame
        if len(chunk) < CHUNK_SIZE:
            chunk = chunk + [chunk[-1]] * (CHUNK_SIZE - len(chunk))

        actions = await request_actions(ws, chunk)
        actions = actions.squeeze(0)
        actions = actions * action_std + action_mean

        real_steps = end - start
        for i in range(real_steps):
            obs = env.get_obs()
            for cam_name, cam_data in obs["observation"].items():
                if "rgb" in cam_data:
                    cam_frames.setdefault(cam_name, []).append(cam_data["rgb"])
            env.take_action(actions[i], action_type="qpos")
            step_count += 1
            if env.check_success():
                success = True

            elapsed = time.time() - t_start
            status = "OK!" if success else "..."
            sys.stdout.write(
                f"\r    Step {step_count:>4d}/{T}  "
                f"{'█' * (step_count * 20 // T)}{'░' * (20 - step_count * 20 // T)}  "
                f"{elapsed:.1f}s  {status}"
            )
            sys.stdout.flush()

        idx += CHUNK_SIZE

    elapsed = time.time() - t_start
    sys.stdout.write(f"\r{' ' * 70}\r")

    env.close_env()

    # Save videos
    os.makedirs(save_dir, exist_ok=True)
    for cam_name, frames in cam_frames.items():
        imageio.mimsave(os.path.join(save_dir, f"{cam_name}.mp4"), frames, fps=30)

    return success, elapsed


# ── Logging helpers ──────────────────────────────────────────────────────────


def print_header(dataset_dir, task_name, server_uri, total_groups, total_episodes, results_dir):
    print()
    print("=" * 80)
    print(f"{'RoboTwin Policy Evaluation':^80}")
    print("=" * 80)
    print(f"  Dataset:    {dataset_dir}")
    print(f"  Task:       {task_name}")
    print(f"  Server:     {server_uri}")
    print(f"  Configs:    {total_groups}")
    print(f"  Episodes:   {total_episodes}")
    print(f"  Results:    {results_dir.absolute()}")


def print_group_header(group_idx, total_groups, group_name, num_episodes):
    print()
    print("-" * 80)
    print(f"Config [{group_idx + 1}/{total_groups}]: {group_name}  ({num_episodes} episodes)")
    print("-" * 80)


def print_episode_result(ep_idx, total_eps, success, elapsed, group_successes, group_total):
    status = " PASS " if success else " FAIL "
    rate = sum(group_successes) / group_total * 100
    bar_width = 20
    filled = int(bar_width * group_total / total_eps)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(
        f"  Episode {ep_idx:>3d}  [{status}]  "
        f"|{bar}|  "
        f"SR: {sum(group_successes)}/{group_total} ({rate:5.1f}%)  "
        f"({elapsed:.1f}s)"
    )


def print_group_summary(group_name, successes, total):
    rate = successes / total * 100 if total > 0 else 0
    print(f"  => {group_name}: {successes}/{total} ({rate:.1f}%)")


def print_final_summary(group_results, total_elapsed):
    print()
    print("=" * 80)
    print(f"{'Final Results':^80}")
    print("=" * 80)
    print()

    total_success = 0
    total_episodes = 0

    print(f"  {'Config':<45} {'Success':>10}  {'Rate':>7}")
    print(f"  {'-' * 45} {'-' * 10}  {'-' * 7}")

    for group_name, (successes, total) in group_results.items():
        rate = successes / total * 100 if total > 0 else 0
        total_success += successes
        total_episodes += total
        display = group_name[:43] + ".." if len(group_name) > 45 else group_name
        print(f"  {display:<45} {successes:>4}/{total:<4}  {rate:>6.1f}%")

    overall_rate = total_success / total_episodes * 100 if total_episodes > 0 else 0
    print(f"  {'-' * 45} {'-' * 10}  {'-' * 7}")
    print(f"  {'OVERALL':<45} {total_success:>4}/{total_episodes:<4}  {overall_rate:>6.1f}%")
    print()
    print(f"  Total time: {total_elapsed:.1f}s")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate policy server on RoboTwin benchmark")
    parser.add_argument("--dataset-dir", required=True, type=str, help="Path to dataset root")
    parser.add_argument("--task-name", type=str, default="adjust_bottle", help="Task name")
    parser.add_argument("--server", type=str, default="ws://localhost:8765", help="Websocket server URL")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    task_name = args.task_name
    server_uri = args.server

    stats = np.load(str(dataset_dir / "action_stats.npz"))
    action_mean = stats["mean"]
    action_std = stats["std"]

    grouped = collect_test_episodes(dataset_dir)
    total_episodes = sum(len(eps) for eps in grouped.values())

    results_dir = orig_root.parent.parent.parent / "results/robotwin"
    print_header(dataset_dir, task_name, server_uri, len(grouped), total_episodes, results_dir)

    group_results = OrderedDict()
    eval_start = time.time()

    for group_idx, (group_name, episodes) in enumerate(grouped.items()):
        print_group_header(group_idx, len(grouped), group_name, len(episodes))

        group_successes = []

        for ep_idx, (ep_path, seed, task_config) in enumerate(episodes):
            ep = Path(ep_path)
            save_dir_tmp = results_dir / group_name / ep.stem

            async with websockets.connect(
                server_uri, max_size=100 * 1024 * 1024,
                ping_interval=None, ping_timeout=None,
            ) as ws:
                success, elapsed = await evaluate_episode(
                    ws, action_mean, action_std,
                    ep_path, seed, task_config, task_name, save_dir_tmp,
                )

            # Rename video files with result tag
            for video_file in save_dir_tmp.glob("*.mp4"):
                tag = "success" if success else "fail"
                new_name = video_file.with_stem(f"{video_file.stem}_{tag}")
                video_file.rename(new_name)

            group_successes.append(success)
            print_episode_result(
                ep_idx, len(episodes), success, elapsed,
                group_successes, ep_idx + 1,
            )

        n_success = sum(group_successes)
        print_group_summary(group_name, n_success, len(group_successes))
        group_results[group_name] = (n_success, len(group_successes))

    total_elapsed = time.time() - eval_start
    print_final_summary(group_results, total_elapsed)


if __name__ == "__main__":
    asyncio.run(main())
