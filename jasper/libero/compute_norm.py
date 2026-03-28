"""
Compute mean and std for state and action across all demos in a dataset directory.

Saves a .npz file in the dataset directory with keys:
    state_mean, state_std, action_mean, action_std

Usage:
    python -m jasper.libero.compute_norm --dataset-dir /home/ubuntu/workspace/LIBERO/libero/datasets/libero_90
"""

import argparse
import os
from glob import glob

import h5py
import numpy as np


def compute_norm(dataset_dir):
    hdf5_paths = sorted(glob(os.path.join(dataset_dir, "*.hdf5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"No HDF5 files found in {dataset_dir}")

    all_states = []
    all_actions = []

    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            for demo_key in sorted(f["data"].keys()):
                grp = f[f"data/{demo_key}"]
                joint = grp["obs/joint_states"][()]
                gripper = grp["obs/gripper_states"][()]
                state = np.concatenate([joint, gripper], axis=1)
                action = grp["actions"][()]
                all_states.append(state)
                all_actions.append(action)

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    norm = {
        "state_mean": all_states.mean(axis=0).astype(np.float32),
        "state_std": all_states.std(axis=0).astype(np.float32),
        "action_mean": all_actions.mean(axis=0).astype(np.float32),
        "action_std": all_actions.std(axis=0).astype(np.float32),
    }

    # Clamp std to avoid division by zero
    norm["state_std"] = np.maximum(norm["state_std"], 1e-6)
    norm["action_std"] = np.maximum(norm["action_std"], 1e-6)

    out_path = os.path.join(dataset_dir, "norm_stats.npz")
    np.savez(out_path, **norm)

    print(f"Computed over {all_states.shape[0]} timesteps from {len(hdf5_paths)} tasks")
    print(f"  state_mean:  {norm['state_mean']}")
    print(f"  state_std:   {norm['state_std']}")
    print(f"  action_mean: {norm['action_mean']}")
    print(f"  action_std:  {norm['action_std']}")
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute normalization stats for LIBERO dataset")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to dataset folder")
    args = parser.parse_args()
    compute_norm(args.dataset_dir)


if __name__ == "__main__":
    main()