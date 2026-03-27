import os
import sys
import glob
import h5py
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

robowin_root = Path("/home/ubuntu/workspace/RoboTwin")
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))
os.chdir(robowin_root)


def decode_rgb(buf):
    """Decode a JPEG/PNG byte buffer to a BGR numpy array."""
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


class RoboTwinDataset(Dataset):
    """Lazy PyTorch dataset over RoboTwin HDF5 episodes.

    Requires prepare_dataset.py to be run first to generate:
        - {subdir}/train.txt and {subdir}/test.txt split files
        - action_stats.npz in the dataset root

    Each index returns a chunk of consecutive frames within one episode:
        cameras: dict of camera_name -> (3, chunk_size, 256, 256) float32 tensor (vjepa2 processed)
        action:  (chunk_size, 14) float32 tensor — normalized absolute joint angles
    """

    def __init__(self, dataset_dir, split="train", chunk_size=30, camera_names=None):
        """
        Args:
            dataset_dir: path to dataset root (contains subfolders like aloha-agilex_clean_50/)
            split: "train" or "test"
            chunk_size: number of consecutive frames per sample
            camera_names: list of camera names to load, or None for all
        """
        assert split in (
            "train",
            "test",
        ), f"split must be 'train' or 'test', got '{split}'"
        self.chunk_size = chunk_size
        self.processor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_preprocessor"
        )

        # Load action normalization stats (always from train)
        stats_path = os.path.join(dataset_dir, "action_stats.npz")
        assert os.path.exists(
            stats_path
        ), f"{stats_path} not found. Run prepare_dataset.py first."
        stats = np.load(stats_path)
        self.action_mean = torch.from_numpy(stats["mean"])
        self.action_std = torch.from_numpy(stats["std"])

        # Collect episode paths from split files
        subdirs = sorted(
            d
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        )

        episode_paths = []
        for subdir in subdirs:
            split_file = os.path.join(dataset_dir, subdir, f"{split}.txt")
            if not os.path.exists(split_file):
                continue
            with open(split_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        episode_paths.append(os.path.join(dataset_dir, subdir, line))

        assert len(episode_paths) > 0, f"No episodes found for split '{split}'"

        # Build chunk index
        self._episode_paths = []
        self._episode_lengths = []
        self._valid_starts = []

        total_chunks = 0
        for path in episode_paths:
            with h5py.File(path, "r") as f:
                T = f["joint_action/vector"].shape[0]
                if camera_names is None:
                    camera_names = sorted(
                        k
                        for k in f["observation"]
                        if f[f"observation/{k}/rgb"].dtype.kind == "S"
                    )

            n_chunks = max(0, T - chunk_size + 1)
            if n_chunks == 0:
                continue
            self._episode_paths.append(path)
            self._episode_lengths.append(T)
            total_chunks += n_chunks
            self._valid_starts.append(total_chunks)

        self._valid_starts = np.array(self._valid_starts)
        self.camera_names = camera_names
        self._len = total_chunks


    def __len__(self):
        return self._len

    def _locate(self, idx):
        """Map a global chunk index to (episode_index, start_frame)."""
        ep_idx = np.searchsorted(self._valid_starts, idx, side="right")
        start = idx if ep_idx == 0 else idx - self._valid_starts[ep_idx - 1]
        return ep_idx, start

    def __getitem__(self, idx):
        ep_idx, start = self._locate(idx)
        end = start + self.chunk_size
        path = self._episode_paths[ep_idx]

        with h5py.File(path, "r") as f:
            actions = f["joint_action/vector"][start:end].astype(np.float32)

            cameras = {}
            for cam_name in self.camera_names:
                frames = []
                for i in range(start, end):
                    buf = f[f"observation/{cam_name}/rgb"][i]
                    frames.append(decode_rgb(buf))
                # processor takes list of HWC uint8, returns [(C, T, H, W)]
                cameras[cam_name] = self.processor(frames)[0]

        action = (torch.from_numpy(actions) - self.action_mean) / self.action_std

        return {
            "cameras": cameras,
            "action": action,
        }


if __name__ == "__main__":
    train_ds = RoboTwinDataset("./dataset", split="train", chunk_size=30)
    test_ds = RoboTwinDataset("./dataset", split="test", chunk_size=30)
    print(f"\nTrain chunks: {len(train_ds)}, Test chunks: {len(test_ds)}")

    sample = train_ds[0]
    print(f"\nAction: shape={sample['action'].shape}, dtype={sample['action'].dtype}")
    for cam_name, img in sample["cameras"].items():
        print(f"{cam_name}: shape={img.shape}, dtype={img.dtype}")

    # Quick DataLoader test
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\nBatch action: {batch['action'].shape}")
    for cam_name in batch["cameras"]:
        print(f"Batch {cam_name}: {batch['cameras'][cam_name].shape}")
