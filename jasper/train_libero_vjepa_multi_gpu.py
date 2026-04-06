"""
Multi-GPU training for Jasper on LIBERO dataset using DDP.

Launch with:
    torchrun --nproc_per_node=8 -m jasper.train_libero_multi_gpu
"""

import os
import copy
import torch
import torch.distributed as dist
import time
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from .jasper import Jasper, JasperConfig
from .libero.dataset_vjepa import LiberoVJEPADataset


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def dump_config(config, ckpt_dir):
    with open(str(ckpt_dir / "config.json"), "w") as f:
        json.dump(config.__dict__, f)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.lerp_(p, 1.0 - decay)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")
    args = parser.parse_args()

    # ---- DDP setup ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    is_main = rank == 0

    # ---- Hyperparameters ----
    dataset_dir = "/home/ubuntu/workspace/LIBERO/libero/datasets/libero_90"
    norm_stats_path = (
        "/home/ubuntu/workspace/LIBERO/libero/datasets/libero_90/norm_stats.npz"
    )
    ckpt_dir = Path("./ckpts/libero/vjepa")

    # Batch: 32/gpu × 8 gpus = 256 effective.
    # 256 is a well-tested batch size for diffusion transformers (DiT, SD3, etc).
    # V-JEPA2 gigantic + 16-layer decoder fits comfortably at 32/gpu on H100 80GB.
    per_gpu_batch_size = 32
    effective_batch_size = per_gpu_batch_size * world_size

    amp_dtype = torch.bfloat16

    # LR: 1e-4 is the standard for AdamW + diffusion transformers at batch ~256.
    # No need for linear scaling — 256 is already the "reference" batch size
    # used by most diffusion training papers (DiT, SD3, flow matching).
    lr = 1e-4
    min_lr = 1e-6  # cosine decays to this, not zero — avoids dead training at the end

    # Weight decay: 0.05 is standard for transformer pretraining (GPT, ViT, DiT).
    # 1e-3 is too low to meaningfully regularize a 630M param model.
    weight_decay = 0.05

    max_grad_norm = 1.0

    # Warmup: 2000 steps at batch 256 lets the model see ~500K samples
    # before hitting peak LR. Prevents early instability with fresh random weights.
    warmup_steps = 2000

    # Total training: 200K steps × 256 batch = 51.2M samples (~81 epochs).
    # Each sample seen with different (noise, timestep) each time, so not redundant.
    # Flow matching converges ~2x faster than DDPM; 80 epochs is a good starting point.
    max_steps = 200_000
    save_every = 10_000
    log_every = 50
    keep_last_n_resumes = 2  # resume checkpoints are ~19GB each, keep only last 2

    # EMA: standard for diffusion models. Use the EMA weights for inference.
    # 0.9999 with 100K steps means the EMA sees an effective window of ~10K steps.
    ema_decay = 0.9999

    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(f"./logs/jasper-libero/{time.time()}")
        print(f"World size: {world_size}")
        print(f"Per-GPU batch size: {per_gpu_batch_size}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"LR: {lr:.2e} -> {min_lr:.2e}")
        print(f"Max steps: {max_steps}")

    # ---- Model ----
    config = JasperConfig(
        device=str(device),
        dtype="float32",
        action_dim=7,
        action_horizon=10,
        hidden_dim=768,
        num_heads=12,
        head_dim=64,
        ff_dim=2048,
        depth=12,
        attn_dropout=0.0,  # no attn dropout — hurts diffusion quality more than it helps
        dropout=0.0,       # regularize via weight decay + EMA instead of dropout
        vjepa2_model="vjepa2_1_vit_large_384",
    )
    model = Jasper(config).to(device)
    torch.set_float32_matmul_precision("high")
    model.vision_encoder = torch.compile(model.vision_encoder, mode="max-autotune")

    # EMA model lives on rank 0 only to save memory on other ranks
    if is_main:
        ema_model = copy.deepcopy(model)
        ema_model.requires_grad_(False)
        ema_model.eval()

    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    scaler = torch.amp.GradScaler(enabled=False)

    # ---- Optimizer & Scheduler ----
    # Only pass trainable params — avoids AdamW allocating states for frozen encoder backbone
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps - warmup_steps, eta_min=min_lr
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # ---- Resume from checkpoint ----
    start_step = 0
    if args.resume is not None:
        if is_main:
            print(f"Resuming from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(resume_ckpt["model"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        scaler.load_state_dict(resume_ckpt["scaler"])
        start_step = resume_ckpt["step"] + 1
        epoch = resume_ckpt["epoch"]
        if is_main:
            ema_model.load_state_dict(resume_ckpt["ema"])
            print(f"Resumed at step {start_step}")
        del resume_ckpt

    # ---- Dataset & Dataloader ----
    dataset = LiberoVJEPADataset(
        dataset_dir=dataset_dir,
        chunk_size=config.action_horizon,
        norm_stats_path=norm_stats_path,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    dl_iter = cycle(dataloader)
    if args.resume is None:
        epoch = 0

    # ---- Training ----
    model.train()
    if is_main:
        dump_config(config, ckpt_dir)

    sampler.set_epoch(epoch)
    step_start = time.perf_counter()
    for step in range(start_step, max_steps):
        if step > start_step and step % len(dataloader) == 0:
            epoch += 1
            sampler.set_epoch(epoch)

        batch = next(dl_iter)

        head = batch["agentview"].to(device, non_blocking=True)
        wrist = batch["eye_in_hand"].to(device, non_blocking=True)
        images = torch.stack([head, wrist], dim=1)
        action = batch["action"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            loss = model(images, action)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if is_main:
            update_ema(ema_model, model.module, ema_decay)

        if is_main and (step + 1) % log_every == 0:
            step_time = (time.perf_counter() - step_start) / log_every
            samples_per_sec = effective_batch_size / step_time
            remaining_steps = max_steps - step - 1
            eta_hours = (remaining_steps * step_time) / 3600

            writer.add_scalar("Loss/Train", loss.item(), step)
            writer.add_scalar("GradNorm", grad_norm.item(), step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], step)
            writer.add_scalar("Perf/StepTime", step_time, step)
            writer.add_scalar("Perf/SamplesPerSec", samples_per_sec, step)

            print(
                f"Step: {step:6d} | Loss: {loss.item():.6f} | GradNorm: {grad_norm.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | {step_time:.2f}s/step | "
                f"{samples_per_sec:.0f} samples/s | ETA: {eta_hours:.1f}h"
            )

            # Log memory once on first log
            if step + 1 == log_every:
                mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
                mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9
                print(f"GPU {local_rank} peak memory: {mem_alloc:.1f}GB allocated, {mem_reserved:.1f}GB reserved")

            step_start = time.perf_counter()

        if (step + 1) % save_every == 0 and is_main:
            # EMA checkpoint for inference (~6.9 GB each, keep all)
            torch.save(
                ema_model.state_dict(),
                str(ckpt_dir / f"checkpoint_{step + 1}_ema.pt"),
            )

            # Full resume checkpoint (~19 GB each, keep last N)
            resume_path = ckpt_dir / f"resume_{step + 1}.pt"
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                str(resume_path),
            )

            # Delete old resume checkpoints beyond keep_last_n_resumes
            resume_files = sorted(ckpt_dir.glob("resume_*.pt"))
            for old_resume in resume_files[:-keep_last_n_resumes]:
                old_resume.unlink()
                print(f"Deleted old resume checkpoint: {old_resume.name}")

    # ---- Cleanup ----
    del dl_iter
    if is_main:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
