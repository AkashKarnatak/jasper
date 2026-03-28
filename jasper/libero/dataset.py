"""
PyTorch Dataset for LIBERO demonstration data.

With chunk_size=1 (default), each sample returns:
    - agentview_rgb:   (3, 128, 128) float32 tensor, normalized to [0, 1]
    - eye_in_hand_rgb: (3, 128, 128) float32 tensor, normalized to [0, 1]
    - state:           (9,) float32 tensor  [joint_states(7), gripper_states(2)]
    - action:          (7,) float32 tensor

With chunk_size=K, each sample returns a chunk of K consecutive frames:
    - agentview_rgb:   (K, 3, 128, 128)
    - eye_in_hand_rgb: (K, 3, 128, 128)
    - state:           (K, 9)
    - action:          (K, 7)

If norm_stats_path is provided, state and action are z-normalized: (x - mean) / std.
Compute norm stats first with: python compute_norm.py --dataset-dir libero/datasets/libero_90

Usage:
    from libero_dataset import LiberoDataset

    # Without normalization
    train_dataset = LiberoDataset("libero/datasets/libero_90")

    # With normalization (use training stats for both train and test)
    train_dataset = LiberoDataset("libero/datasets/libero_90", norm_stats_path="libero/datasets/libero_90/norm_stats.npz")
    test_dataset  = LiberoDataset("libero/datasets/libero_10", norm_stats_path="libero/datasets/libero_90/norm_stats.npz")

    sample = train_dataset[0]
    # sample["agentview_rgb"].shape    -> (3, 128, 128)
    # sample["eye_in_hand_rgb"].shape  -> (3, 128, 128)
    # sample["state"].shape            -> (9,)
    # sample["action"].shape           -> (7,)
    # sample["task_id"]                -> int
    # sample["task_name"]              -> str
"""

import bisect
import os
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LiberoDataset(Dataset):
    def __init__(self, dataset_dir, norm_stats_path=None, chunk_size=1, use_vjepa2=False, transform=None):
        """
        Args:
            dataset_dir: Path to a folder of HDF5 files (e.g. "libero/datasets/libero_90")
            norm_stats_path: Path to norm_stats.npz (from compute_norm.py). If provided,
                             state and action are z-normalized. Use training set stats for
                             both train and test.
            chunk_size: Number of consecutive frames per sample. Only starting indices
                        with enough remaining frames in the episode are included.
            use_vjepa2: If True, process camera observations with the VJEPA2 preprocessor.
                        Output images will be (3, K, 256, 256) instead of (K, 3, 128, 128).
                        The transform parameter is ignored when this is True.
            transform: Optional transform applied to image tensors (both cameras).
                       Ignored when use_vjepa2=True.
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.chunk_size = chunk_size
        self.use_vjepa2 = use_vjepa2

        if use_vjepa2:
            self.processor = torch.hub.load(
                "facebookresearch/vjepa2", "vjepa2_preprocessor"
            )

        # Load normalization stats
        if norm_stats_path is not None:
            stats = np.load(norm_stats_path)
            self._state_mean = stats["state_mean"]
            self._state_std = stats["state_std"]
            self._action_mean = stats["action_mean"]
            self._action_std = stats["action_std"]
            self._normalize = True
        else:
            self._normalize = False

        # Build a compact index: one entry per demo, use cumulative sums + bisect
        # to map a flat index to (demo, timestep) at query time.
        self._demos = []       # list of (hdf5_path, task_id, task_name, demo_key)
        self._cum_lengths = [] # cumulative valid timesteps (for bisect)
        self._hdf5_paths = sorted(glob(os.path.join(dataset_dir, "*.hdf5")))

        if len(self._hdf5_paths) == 0:
            raise FileNotFoundError(f"No HDF5 files found in {dataset_dir}")

        total = 0
        for task_id, hdf5_path in enumerate(self._hdf5_paths):
            task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
            with h5py.File(hdf5_path, "r") as f:
                for demo_key in sorted(f["data"].keys()):
                    num_steps = f[f"data/{demo_key}/actions"].shape[0]
                    valid = num_steps - chunk_size + 1
                    if valid <= 0:
                        continue
                    self._demos.append((hdf5_path, task_id, task_name, demo_key))
                    total += valid
                    self._cum_lengths.append(total)

        self._total_len = total

        # Cache for open HDF5 file handles (opened lazily)
        self._h5_cache = {}

        print(
            f"LiberoDataset: {len(self._hdf5_paths)} tasks, "
            f"{len(self._demos)} demos, "
            f"{self._total_len} total samples from {dataset_dir}"
        )

    def _get_h5(self, path):
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r")
        return self._h5_cache[path]

    def __len__(self):
        return self._total_len

    def _resolve_idx(self, idx):
        """Map a flat index to (demo_index, timestep) via binary search."""
        demo_idx = bisect.bisect_right(self._cum_lengths, idx)
        t = idx - (self._cum_lengths[demo_idx - 1] if demo_idx > 0 else 0)
        return demo_idx, t

    def __getitem__(self, idx):
        demo_idx, t = self._resolve_idx(idx)
        hdf5_path, task_id, task_name, demo_key = self._demos[demo_idx]
        f = self._get_h5(hdf5_path)
        grp = f[f"data/{demo_key}"]
        s = slice(t, t + self.chunk_size)

        # Raw images: (K, H, W, 3) uint8 — stored vertically flipped in LIBERO HDF5
        agentview_raw = grp["obs/agentview_rgb"][s][:, ::-1].copy()
        eye_in_hand_raw = grp["obs/eye_in_hand_rgb"][s][:, ::-1].copy()

        if self.use_vjepa2:
            # VJEPA2 preprocessor expects list of (H, W, 3) uint8 numpy frames
            # and returns [(3, K, crop_size, crop_size)] float32 ImageNet-normalized
            agentview = self.processor(list(agentview_raw))[0]
            eye_in_hand = self.processor(list(eye_in_hand_raw))[0]
        else:
            # (K, H, W, 3) uint8 -> (K, 3, H, W) float32 [0, 1]
            agentview = torch.from_numpy(
                agentview_raw.astype(np.float32) / 255.0
            ).permute(0, 3, 1, 2)
            eye_in_hand = torch.from_numpy(
                eye_in_hand_raw.astype(np.float32) / 255.0
            ).permute(0, 3, 1, 2)

            if self.transform is not None:
                agentview = torch.stack([self.transform(img) for img in agentview])
                eye_in_hand = torch.stack([self.transform(img) for img in eye_in_hand])

        # State: [joint_states(7), gripper_states(2)] -> (K, 9)
        joint = grp["obs/joint_states"][s]
        gripper = grp["obs/gripper_states"][s]
        state = np.concatenate([joint, gripper], axis=1).astype(np.float32)

        # Action: (K, 7)
        action = grp["actions"][s].astype(np.float32)

        if self._normalize:
            state = (state - self._state_mean) / self._state_std
            action = (action - self._action_mean) / self._action_std

        state = torch.from_numpy(state)
        action = torch.from_numpy(action)

        return {
            "agentview_rgb": agentview,
            "eye_in_hand_rgb": eye_in_hand,
            "state": state,
            "action": action,
            "task_id": task_id,
            "task_name": task_name,
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

    @property
    def task_names(self):
        return [
            os.path.basename(p).replace("_demo.hdf5", "") for p in self._hdf5_paths
        ]


if __name__ == "__main__":
    dataset_dir = "/home/ubuntu/workspace/LIBERO/libero/datasets/libero_90"
    norm_stats = "/home/ubuntu/workspace/LIBERO/libero/datasets/libero_90/norm_stats.npz"

    for chunk in [1, 10]:
        print(f"\n{'='*50}")
        print(f"chunk_size={chunk}")
        print(f"{'='*50}")
        ds = LiberoDataset(dataset_dir, norm_stats_path=norm_stats, chunk_size=chunk, use_vjepa2=True)
        sample = ds[0]
        print(f"agentview_rgb:   {sample['agentview_rgb'].shape}")
        print(f"eye_in_hand_rgb: {sample['eye_in_hand_rgb'].shape}")
        print(f"state:           {sample['state'].shape}")
        print(f"action:          {sample['action'].shape}")
        print(f"task_id:         {sample['task_id']}")
        print(f"task_name:       {sample['task_name']}")
        print(f"Total samples:   {len(ds)}")
        ds.close()
