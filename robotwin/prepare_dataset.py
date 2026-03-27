"""Generate train/test splits and compute action normalization stats.

Usage:
    python prepare_dataset.py --dataset_dir ./dataset --test_ratio 0.1

Creates per-subfolder:
    train.txt  — relative paths to train episodes
    test.txt   — relative paths to test episodes

Creates in dataset root:
    action_stats.npz  — action mean and std computed from train episodes only
"""

import os
import glob
import argparse
import hashlib
import numpy as np
import h5py


def is_test_episode(path, test_ratio):
    h = int(hashlib.md5(path.encode()).hexdigest(), 16)
    return (h % 1000) < int(test_ratio * 1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    subdirs = sorted(
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    assert len(subdirs) > 0, f"No subdirectories found in {dataset_dir}"

    all_train_paths = []

    for subdir in subdirs:
        data_dir = os.path.join(dataset_dir, subdir, "data")
        if not os.path.isdir(data_dir):
            continue

        episodes = sorted(glob.glob(os.path.join(data_dir, "episode*.hdf5")))
        train, test = [], []
        for ep in episodes:
            rel = os.path.relpath(ep, os.path.join(dataset_dir, subdir))
            if is_test_episode(ep, args.test_ratio):
                test.append(rel)
            else:
                train.append(rel)
                all_train_paths.append(ep)

        train_path = os.path.join(dataset_dir, subdir, "train.txt")
        test_path = os.path.join(dataset_dir, subdir, "test.txt")

        with open(train_path, "w") as f:
            f.write("\n".join(train) + "\n")
        with open(test_path, "w") as f:
            f.write("\n".join(test) + "\n")

        print(f"{subdir}: {len(train)} train, {len(test)} test")

    # Compute action stats from train episodes only (batch Welford's)
    print(f"\nComputing action stats from {len(all_train_paths)} train episodes...")
    n = 0
    mean = np.zeros(14, dtype=np.float64)
    m2 = np.zeros(14, dtype=np.float64)

    for path in all_train_paths:
        with h5py.File(path, "r") as f:
            actions = f["joint_action/vector"][:]
            T = actions.shape[0]
            batch_mean = actions.mean(axis=0)
            batch_var = actions.var(axis=0)
            if n == 0:
                mean = batch_mean
                m2 = batch_var * T
            else:
                delta = batch_mean - mean
                new_n = n + T
                mean = (mean * n + batch_mean * T) / new_n
                m2 += batch_var * T + delta**2 * (n * T / new_n)
            n += T

    std = np.sqrt(m2 / n).clip(min=1e-6)

    stats_path = os.path.join(dataset_dir, "action_stats.npz")
    np.savez(stats_path, mean=mean.astype(np.float32), std=std.astype(np.float32))

    print(f"Action mean: {mean}")
    print(f"Action std:  {std}")
    print(f"Saved to {stats_path}")


if __name__ == "__main__":
    main()
