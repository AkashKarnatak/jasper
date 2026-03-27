"""
Evaluate a policy server on all episodes across all tasks in a LIBERO dataset directory.

For each HDF5 file in the directory, runs every demo episode through the policy server
and reports per-task and overall success rates.

Usage:
    python jasper/libero/run_evals.py --dataset-dir /home/ubuntu/workspace/LIBERO/libero/datasets/libero_10 \
                        --norm-stats /home/ubuntu/workspace/LIBERO/libero/datasets/libero_90/norm_stats.npz

    python jasper/libero/run_evals.py --dataset-dir /home/ubuntu/workspace/LIBERO/libero/datasets/libero_10 \
                        --norm-stats /home/ubuntu/workspace/LIBERO/libero/datasets/libero_90/norm_stats.npz \
                        --server ws://localhost:8765 \
                        --max-episodes 5
"""

import argparse
import asyncio
import json
import math
import os
import xml.etree.ElementTree as ET
from glob import glob

import sys
import time

import h5py
import imageio
import msgpack
import numpy as np
import robosuite
import websockets
from pathlib import Path

orig_root = Path(__file__).resolve()
sim_root = "/home/ubuntu/workspace/LIBERO"
if sim_root not in sys.path:
    sys.path.insert(0, sim_root)
os.chdir(sim_root)

import libero.libero.utils.utils as libero_utils
from libero.libero.envs import TASK_MAPPING

_SCRIPT_DIR = os.path.abspath(os.curdir)
_ROBOSUITE_PATH = os.path.split(robosuite.__file__)[0]
_LIBERO_ASSETS_PATH = os.path.join(_SCRIPT_DIR, "libero", "libero")

CHUNK_SIZE = 10


def postprocess_model_xml(xml_str):
    tree = ET.fromstring(xml_str)
    asset = tree.find("asset")
    for elem in asset.findall("mesh") + asset.findall("texture"):
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_parts = old_path.split("/")
        if "robosuite" in old_parts:
            idx = max(i for i, v in enumerate(old_parts) if v == "robosuite")
            elem.set("file", os.path.join(_ROBOSUITE_PATH, *old_parts[idx + 1 :]))
        elif "chiliocosm" in old_parts:
            idx = old_parts.index("chiliocosm")
            elem.set("file", os.path.join(_LIBERO_ASSETS_PATH, *old_parts[idx + 1 :]))
        elif "libero" in old_parts:
            idx = max(i for i, v in enumerate(old_parts) if v == "libero")
            elem.set("file", os.path.join(_SCRIPT_DIR, "libero", *old_parts[idx + 1 :]))
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def create_env_from_hdf5(f):
    env_args = json.loads(f["data"].attrs["env_args"])
    problem_name = env_args["problem_name"]
    env_kwargs = env_args["env_kwargs"]
    bddl_file_name = f["data"].attrs["bddl_file_name"]
    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=False,
        camera_names=["robot0_eye_in_hand", "agentview"],
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=None,
    )
    env = TASK_MAPPING[problem_name](**env_kwargs)
    return env


def serialize_frame(frame):
    return {
        "data": frame.tobytes(),
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
    }


def deserialize_actions(actions_msg):
    return np.frombuffer(actions_msg["data"], dtype=actions_msg["dtype"]).reshape(
        actions_msg["shape"]
    )


def save_video(frames, path, fps=20):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


async def evaluate_episode(ws, env, f, ep_key, action_mean, action_std, video_path):
    """Evaluate a single episode. Returns True if the task was completed."""
    agentview_obs = f[f"data/{ep_key}/obs/agentview_rgb"][()]
    num_frames = agentview_obs.shape[0]
    num_chunks = math.ceil(num_frames / CHUNK_SIZE)

    model_xml = f[f"data/{ep_key}"].attrs["model_file"]
    states = f[f"data/{ep_key}/states"][()]

    # Reset env to recorded initial state
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue

    model_xml = postprocess_model_xml(model_xml)
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    task_success = False
    video_frames = []
    step_count = 0
    t_start = time.time()

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, num_frames)
        chunk = agentview_obs[start:end]

        # Pad last chunk
        if len(chunk) < CHUNK_SIZE:
            pad_count = CHUNK_SIZE - len(chunk)
            padding = np.stack([chunk[-1]] * pad_count, axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)

        frames_msg = [serialize_frame(chunk[i]) for i in range(CHUNK_SIZE)]
        request = msgpack.packb({"frames": frames_msg})
        await ws.send(request)

        response = msgpack.unpackb(await ws.recv(), raw=False)
        if "error" in response:
            print(f"\r    Server error: {response['error']}")
            break

        actions = deserialize_actions(response["actions"]).squeeze(0)
        actions = actions * action_std + action_mean

        real_steps = end - start
        for i in range(real_steps):
            obs, _, _, _ = env.step(actions[i])
            step_count += 1

            agentview_frame = obs["agentview_image"][::-1]
            wrist_frame = obs["robot0_eye_in_hand_image"][::-1]
            video_frames.append(np.concatenate([agentview_frame, wrist_frame], axis=1))

            if env._check_success():
                task_success = True

            # Live step progress on same line
            elapsed = time.time() - t_start
            status = "..." if not task_success else "OK!"
            sys.stdout.write(
                f"\r    Step {step_count:>4d}/{num_frames}  "
                f"[{'=' * (step_count * 20 // num_frames):<20s}]  "
                f"{elapsed:.1f}s  {status}"
            )
            sys.stdout.flush()

    elapsed = time.time() - t_start
    sys.stdout.write(f"\r{' ' * 70}\r")  # clear the progress line

    # Save video
    save_video(video_frames, video_path)

    return task_success, elapsed


def print_header():
    print()
    print("=" * 80)
    print(f"{'LIBERO Policy Evaluation':^80}")
    print("=" * 80)


def print_task_header(task_idx, total_tasks, task_name):
    print()
    print("-" * 80)
    print(f"Task [{task_idx + 1}/{total_tasks}]: {task_name}")
    print("-" * 80)


def print_progress(ep_idx, total_eps, success, elapsed, task_successes, task_total):
    status = " PASS " if success else " FAIL "
    rate = sum(task_successes) / task_total * 100
    bar_width = 20
    filled = int(bar_width * task_total / total_eps)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(
        f"  Episode {ep_idx:>3d}  [{status}]  "
        f"|{bar}|  "
        f"SR: {sum(task_successes)}/{task_total} ({rate:5.1f}%)  "
        f"({elapsed:.1f}s)"
    )


def print_task_summary(task_name, successes, total):
    rate = successes / total * 100 if total > 0 else 0
    print(f"  => {task_name}: {successes}/{total} ({rate:.1f}%)")


def print_final_summary(task_results):
    print()
    print("=" * 80)
    print(f"{'Final Results':^80}")
    print("=" * 80)
    print()

    total_success = 0
    total_episodes = 0

    # Per-task results
    print(f"  {'Task':<55} {'Success':>10}  {'Rate':>7}")
    print(f"  {'─' * 55} {'─' * 10}  {'─' * 7}")

    for task_name, (successes, total) in task_results.items():
        rate = successes / total * 100 if total > 0 else 0
        total_success += successes
        total_episodes += total
        # Truncate long names
        display_name = task_name[:53] + ".." if len(task_name) > 55 else task_name
        print(f"  {display_name:<55} {successes:>4}/{total:<4}  {rate:>6.1f}%")

    # Overall
    overall_rate = total_success / total_episodes * 100 if total_episodes > 0 else 0
    print(f"  {'─' * 55} {'─' * 10}  {'─' * 7}")
    print(
        f"  {'OVERALL':<55} {total_success:>4}/{total_episodes:<4}  {overall_rate:>6.1f}%"
    )
    print()


async def run_evaluation(dataset_dir, norm_stats_path, server_url, max_episodes):
    stats = np.load(norm_stats_path)
    action_mean = stats["action_mean"]
    action_std = stats["action_std"]

    hdf5_paths = sorted(glob(os.path.join(dataset_dir, "*.hdf5")))
    if not hdf5_paths:
        print(f"No HDF5 files found in {dataset_dir}")
        return

    results_dir = orig_root.parent.parent.parent / "results/libero"
    results_dir.mkdir(exist_ok=True, parents=True)

    print_header()
    print(f"  Dataset:    {dataset_dir}")
    print(f"  Norm stats: {norm_stats_path}")
    print(f"  Server:     {server_url}")
    print(f"  Tasks:      {len(hdf5_paths)}")
    print(f"  Results:    {results_dir.absolute()}/")
    if max_episodes:
        print(f"  Max eps:    {max_episodes} per task")

    task_results = {}
    eval_start = time.time()

    async with websockets.connect(server_url, max_size=100 * 1024 * 1024) as ws:
        for task_idx, hdf5_path in enumerate(hdf5_paths):
            task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
            f = h5py.File(hdf5_path, "r")

            demo_keys = sorted(f["data"].keys())
            if max_episodes:
                demo_keys = demo_keys[:max_episodes]

            print_task_header(task_idx, len(hdf5_paths), task_name)

            env = create_env_from_hdf5(f)
            task_successes = []

            task_video_dir = results_dir / task_name

            for ep_idx, demo_key in enumerate(demo_keys):
                # Save to temp path first, rename with result after evaluation
                video_path_tmp = task_video_dir / f"{demo_key}_tmp.mp4"
                success, elapsed = await evaluate_episode(
                    ws, env, f, demo_key, action_mean, action_std, str(video_path_tmp)
                )
                tag = "success" if success else "fail"
                video_path_final = task_video_dir / f"{demo_key}_{tag}.mp4"
                os.rename(video_path_tmp, str(video_path_final))
                task_successes.append(success)
                print_progress(
                    ep_idx, len(demo_keys), success, elapsed, task_successes, ep_idx + 1
                )

            env.close()
            f.close()

            n_success = sum(task_successes)
            print_task_summary(task_name, n_success, len(task_successes))
            task_results[task_name] = (n_success, len(task_successes))

    total_elapsed = time.time() - eval_start
    print_final_summary(task_results)
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Videos saved to: {results_dir.absolute()}/")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate policy server on LIBERO benchmark"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset folder (e.g. libero/datasets/libero_10)",
    )
    parser.add_argument(
        "--norm-stats", type=str, required=True, help="Path to norm_stats.npz"
    )
    parser.add_argument(
        "--server", type=str, default="ws://localhost:8765", help="Websocket server URL"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Max episodes per task (default: all)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            args.dataset_dir, args.norm_stats, args.server, args.max_episodes
        )
    )


if __name__ == "__main__":
    main()
