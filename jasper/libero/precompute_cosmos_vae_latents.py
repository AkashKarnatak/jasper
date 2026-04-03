"""
Precompute Cosmos VAE latents for all LIBERO demos.

Encodes each demo's full agentview and wrist video through the Cosmos 3D VAE
and saves latents per-demo. During training, temporal chunks are sliced from
the precomputed full-episode latents.

Output structure:
    latent_dir/
        metadata.json
        task_000/
            demo_0.pt    # {"agentview": (z, lt, lh, lw), "wrist": (z, lt, lh, lw), ...}
            demo_1.pt
        task_001/
            ...

Usage:
    python -m jasper.libero.precompute_vae_latents \
        --dataset-dir /home/ubuntu/workspace/LIBERO/libero/datasets/libero_90 \
        --output-dir /home/ubuntu/workspace/LIBERO/libero/datasets/libero_90_vae_latents
"""

import argparse
import json
import math
import os
import time
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm
from diffusers import Cosmos2VideoToWorldPipeline


def pad_temporal(frames_tensor, factor):
    """Pad temporal dim to a multiple of factor by repeating last frame."""
    t = frames_tensor.shape[2]
    remainder = t % factor
    if remainder == 0:
        return frames_tensor, 0
    pad_t = factor - remainder
    padding = frames_tensor[:, :, -1:].expand(-1, -1, pad_t, -1, -1)
    return torch.cat([frames_tensor, padding], dim=2), pad_t


def encode_video(vae, frames_np, temporal_factor, device, dtype):
    """
    Encode a full video through the Cosmos VAE.

    Args:
        frames_np: (T, H, W, 3) uint8, already vertically flipped
    Returns:
        latent: (z_dim, lt, lh, lw) float32 on CPU
        num_padded: number of temporal frames padded
    """
    frames = torch.from_numpy(frames_np.astype(np.float32) / 255.0)
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
    frames = frames * 2.0 - 1.0

    frames, num_padded = pad_temporal(frames, temporal_factor)
    frames = frames.to(device=device, dtype=dtype)

    with torch.no_grad():
        latent = vae.encode(frames).latent_dist.sample()

    return latent.squeeze(0).float().cpu(), num_padded


def format_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    print("Loading Cosmos VAE...")
    pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
        "nvidia/Cosmos-Predict2-2B-Video2World",
    )
    vae = pipeline.vae.to(device=device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)

    temporal_factor = vae.config.scale_factor_temporal
    spatial_factor = vae.config.scale_factor_spatial
    z_dim = vae.config.z_dim

    del pipeline
    torch.cuda.empty_cache()

    print(f"VAE: z_dim={z_dim}, temporal_factor={temporal_factor}, spatial_factor={spatial_factor}")

    # Probe latent dims from a test encode
    test_frames = np.random.randint(0, 255, (temporal_factor, 128, 128, 3), dtype=np.uint8)
    test_latent, _ = encode_video(vae, test_frames, temporal_factor, device, dtype)
    _, lh, lw = test_latent.shape[1:]  # spatial latent dims
    print(f"Test: {temporal_factor} frames -> latent {tuple(test_latent.shape)}")
    del test_frames, test_latent

    # Discover all demos and collect frame counts
    hdf5_paths = sorted(glob(os.path.join(args.dataset_dir, "*.hdf5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"No HDF5 files found in {args.dataset_dir}")

    print(f"\nScanning {len(hdf5_paths)} tasks...")
    demo_info = []  # (hdf5_path, task_id, task_name, demo_key, num_frames)
    total_frames = 0
    for task_id, hdf5_path in enumerate(hdf5_paths):
        task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
        with h5py.File(hdf5_path, "r") as f:
            for demo_key in sorted(f["data"].keys()):
                num_frames = f[f"data/{demo_key}/obs/agentview_rgb"].shape[0]
                demo_info.append((hdf5_path, task_id, task_name, demo_key, num_frames))
                total_frames += num_frames

    # Estimate storage
    # Each demo: 2 views × (z_dim, ceil(T/temporal_factor), lh, lw) × 4 bytes (float32)
    total_latent_frames = sum(
        math.ceil(d[4] / temporal_factor) for d in demo_info
    )
    bytes_per_latent_frame = z_dim * lh * lw * 4  # float32
    estimated_bytes = total_latent_frames * 2 * bytes_per_latent_frame  # 2 views
    # Add ~10% overhead for torch.save serialization
    estimated_bytes = int(estimated_bytes * 1.1)

    print(f"\n{'='*50}")
    print(f"  Tasks:            {len(hdf5_paths)}")
    print(f"  Demos:            {len(demo_info)}")
    print(f"  Total frames:     {total_frames:,}")
    print(f"  Total latent frames (per view): {total_latent_frames:,}")
    print(f"  Latent shape per frame: ({z_dim}, {lh}, {lw})")
    print(f"  Estimated storage: {format_bytes(estimated_bytes)}")
    print(f"{'='*50}\n")

    # Benchmark: encode one demo to estimate speed
    bench_demo = demo_info[0]
    with h5py.File(bench_demo[0], "r") as f:
        bench_frames = f[f"data/{bench_demo[3]}/obs/agentview_rgb"][()][:, ::-1].copy()
    t0 = time.perf_counter()
    encode_video(vae, bench_frames, temporal_factor, device, dtype)
    torch.cuda.synchronize()
    bench_time = time.perf_counter() - t0
    sec_per_frame = bench_time / len(bench_frames)
    total_estimated_sec = total_frames * 2 * sec_per_frame  # 2 views
    del bench_frames

    print(f"Benchmark: {len(demo_info[0][3])} frames in {bench_time:.2f}s ({1/sec_per_frame:.0f} frames/s)")
    print(f"Estimated total time: {format_time(total_estimated_sec)}")
    print(f"  ({total_frames * 2:,} frames × {sec_per_frame*1000:.1f}ms/frame)")
    print()

    # Save metadata
    metadata = {
        "z_dim": z_dim,
        "temporal_factor": temporal_factor,
        "spatial_factor": spatial_factor,
        "encoder_dim": z_dim,
        "latent_spatial_h": lh,
        "latent_spatial_w": lw,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Process all demos
    total_encoded = 0
    total_skipped = 0
    total_bytes_written = 0
    start_time = time.perf_counter()

    current_task_id = None
    for i, (hdf5_path, task_id, task_name, demo_key, num_frames) in enumerate(
        tqdm(demo_info, desc="Encoding")
    ):
        task_dir = output_dir / f"task_{task_id:03d}"
        task_dir.mkdir(exist_ok=True)

        if task_id != current_task_id:
            current_task_id = task_id
            tqdm.write(f"[{task_id+1}/{len(hdf5_paths)}] {task_name}")

        save_path = task_dir / f"{demo_key}.pt"
        if save_path.exists():
            total_skipped += 1
            continue

        with h5py.File(hdf5_path, "r") as f:
            grp = f[f"data/{demo_key}"]
            agentview_frames = grp["obs/agentview_rgb"][()][:, ::-1].copy()
            wrist_frames = grp["obs/eye_in_hand_rgb"][()][:, ::-1].copy()

        agentview_latent, ag_pad = encode_video(
            vae, agentview_frames, temporal_factor, device, dtype
        )
        wrist_latent, _ = encode_video(
            vae, wrist_frames, temporal_factor, device, dtype
        )

        torch.save({
            "agentview": agentview_latent,
            "wrist": wrist_latent,
            "num_frames": num_frames,
            "num_padded": ag_pad,
        }, str(save_path))

        total_bytes_written += save_path.stat().st_size
        total_encoded += 1

        # Print progress every 100 demos
        if (total_encoded) % 100 == 0:
            elapsed = time.perf_counter() - start_time
            demos_remaining = len(demo_info) - i - 1 - total_skipped
            if total_encoded > 0:
                sec_per_demo = elapsed / total_encoded
                eta = demos_remaining * sec_per_demo
                tqdm.write(
                    f"  Progress: {total_encoded} encoded, {total_skipped} skipped | "
                    f"Disk: {format_bytes(total_bytes_written)} | "
                    f"ETA: {format_time(eta)}"
                )

    elapsed = time.perf_counter() - start_time
    print(f"\n{'='*50}")
    print(f"  Encoded:  {total_encoded} demos")
    print(f"  Skipped:  {total_skipped} demos (already exist)")
    print(f"  Disk:     {format_bytes(total_bytes_written)}")
    print(f"  Time:     {format_time(elapsed)}")
    print(f"  Saved to: {output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
