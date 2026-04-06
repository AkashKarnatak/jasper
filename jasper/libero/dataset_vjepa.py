"""
LIBERO dataset for VJEPA2-based training.

Each sample returns a chunk of K consecutive frames:
    - agentview:    (3, K, 256, 256) float32 tensor (VJEPA2 preprocessed)
    - eye_in_hand:  (3, K, 256, 256) float32 tensor (VJEPA2 preprocessed)
    - action:       (K, 7) float32 tensor

If norm_stats_path is provided, actions are z-normalized.
"""

import bisect
import os
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LiberoVJEPADataset(Dataset):
    def __init__(self, dataset_dir, chunk_size, norm_stats_path=None, train=True, train_percent=90):
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size

        self.processor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_preprocessor"
        )

        if norm_stats_path is not None:
            stats = np.load(norm_stats_path)
            self._action_mean = stats["action_mean"]
            self._action_std = stats["action_std"]
            self._normalize = True
        else:
            self._normalize = False

        self._demos = []
        self._cum_lengths = []
        self._hdf5_paths = sorted(glob(os.path.join(dataset_dir, "*.hdf5")))

        if not self._hdf5_paths:
            raise FileNotFoundError(f"No HDF5 files found in {dataset_dir}")

        total = 0
        for task_id, hdf5_path in enumerate(self._hdf5_paths):
            with h5py.File(hdf5_path, "r") as f:
                all_demos = sorted(f["data"].keys())
                split = int(len(all_demos) * train_percent / 100)
                demos = all_demos[:split] if train else all_demos[split:]
                for demo_key in demos:
                    num_steps = f[f"data/{demo_key}/actions"].shape[0]
                    valid = num_steps - chunk_size + 1
                    if valid <= 0:
                        continue
                    self._demos.append((hdf5_path, task_id, demo_key))
                    total += valid
                    self._cum_lengths.append(total)

        self._total_len = total
        self._h5_cache = {}

        split_name = "train" if train else "val"
        print(
            f"LiberoVJEPADataset ({split_name}): {len(self._hdf5_paths)} tasks, "
            f"{len(self._demos)} demos, {self._total_len} samples"
        )

    def _get_h5(self, path):
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r")
        return self._h5_cache[path]

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        demo_idx = bisect.bisect_right(self._cum_lengths, idx)
        t = idx - (self._cum_lengths[demo_idx - 1] if demo_idx > 0 else 0)
        hdf5_path, task_id, demo_key = self._demos[demo_idx]

        f = self._get_h5(hdf5_path)
        grp = f[f"data/{demo_key}"]
        s = slice(t, t + self.chunk_size)

        # (K, H, W, 3) uint8 — stored vertically flipped in LIBERO HDF5
        agentview_raw = grp["obs/agentview_rgb"][s][:, ::-1].copy()
        eye_in_hand_raw = grp["obs/eye_in_hand_rgb"][s][:, ::-1].copy()

        # VJEPA2 preprocessor: list of (H, W, 3) uint8 -> (3, K, 256, 256)
        agentview = self.processor(list(agentview_raw))[0]
        eye_in_hand = self.processor(list(eye_in_hand_raw))[0]

        action = grp["actions"][s].astype(np.float32)
        if self._normalize:
            action = (action - self._action_mean) / self._action_std
        action = torch.from_numpy(action)

        return {
            "agentview": agentview,
            "eye_in_hand": eye_in_hand,
            "action": action,
            "task_id": task_id,
        }

    def close(self):
        for f in self._h5_cache.values():
            f.close()
        self._h5_cache.clear()

    def __del__(self):
        self.close()

    @property
    def num_tasks(self):
        return len(self._hdf5_paths)


if __name__ == "__main__":
    dataset_dir = "/home/ubuntu/workspace/LIBERO/libero/datasets/libero_90"
    norm_stats = "/home/ubuntu/workspace/LIBERO/libero/datasets/libero_90/norm_stats.npz"

    ds = LiberoVJEPADataset(dataset_dir, chunk_size=11, norm_stats_path=norm_stats)
    sample = ds[0]
    print(f"agentview:   {sample['agentview'].shape}")
    print(f"eye_in_hand: {sample['eye_in_hand'].shape}")
    print(f"action:      {sample['action'].shape}")
    print(f"task_id:     {sample['task_id']}")
    print(f"Total:       {len(ds)}")
    ds.close()