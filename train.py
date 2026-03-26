import torch
import time
from jasper import Jasper, JasperConfig

# from robotwin.dataset import RoboTwinDataset
from libero.dataset import LiberoDataset
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


ckpt_dir = Path("./ckpts")
batch_size = 128
dtype = torch.float32
device = "cuda"
lr = (batch_size / 32) * 1e-4
weight_decay = 1e-3
max_grad_norm = 1.0
warmup_steps = 1000
save_every = 4000  // (batch_size // 32)
max_steps = 50_000 // (batch_size // 32)
writer = SummaryWriter(f"./logs/jasper/{time.time()}")

ckpt_dir.mkdir(parents=True, exist_ok=True)

config = JasperConfig(
    dtype=dtype,
    device=device,
    action_horizon=10,
    action_dim=7,
    hidden_dim=1536,
    decoder_num_layers=16,
    ff_dim=4096,
    num_heads=12,
    head_dim=128,
    vjepa2_model="vjepa2_1_vit_large_384"
)
model = Jasper(config).to(dtype).to(device, non_blocking=True)
torch.set_float32_matmul_precision("high")
model.vision_encoder = torch.compile(model.vision_encoder)
# model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-2, total_iters=warmup_steps
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_steps - warmup_steps
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps],
)

# dataset = RoboTwinDataset("./dataset", split="train", chunk_size=config.action_horizon)
dataset = LiberoDataset(
    "/home/ubuntu/workspace/LIBERO/libero/datasets/libero_90",
    norm_stats_path="/home/ubuntu/workspace/LIBERO/libero/datasets/libero_90/norm_stats.npz",
    chunk_size=config.action_horizon,
    use_vjepa2=True,
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
dl_iter = cycle(dataloader)

model.train()

timings = {
    "data": 0.0,
    "transfer": 0.0,
    "forward": 0.0,
    "backward": 0.0,
    "optim": 0.0,
    "total": 0.0,
}

log_every = 1

for step in range(max_steps):
    step_start = time.perf_counter()

    # ---- DATA LOADING (now correctly measured) ----
    t0 = time.perf_counter()
    batch = next(dl_iter)
    t1 = time.perf_counter()

    # ---- TRANSFER ----
    images = batch["agentview_rgb"].to(dtype).to(device, non_blocking=True)
    action = batch["action"].to(dtype).to(device, non_blocking=True)

    print(images.shape)
    print(action.shape)

    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    # ---- FORWARD ----
    loss = model(images, action)

    if device == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    # ---- BACKWARD ----
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    if device == "cuda":
        torch.cuda.synchronize()
    t4 = time.perf_counter()

    # ---- OPTIM ----
    optimizer.step()
    scheduler.step()

    if device == "cuda":
        torch.cuda.synchronize()
    t5 = time.perf_counter()

    # ---- ACCUMULATE ----
    timings["data"] += t1 - t0
    timings["transfer"] += t2 - t1
    timings["forward"] += t3 - t2
    timings["backward"] += t4 - t3
    timings["optim"] += t5 - t4
    timings["total"] += t5 - step_start

    writer.add_scalar("Loss/Train", loss.item(), step)
    print(
        f"Step: {step:6d} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.2e}"
    )

    if (step + 1) % log_every == 0:
        print("\n=== Timing (avg per batch) ===")
        for k in timings:
            print(f"{k:>10}: {timings[k] / log_every:.6f} sec")
        print("================================\n")

        for k in timings:
            timings[k] = 0.0

    if (step + 1) % save_every == 0:
        torch.save(
            model.state_dict(), str(ckpt_dir / f"checkpoint_{step + 1}.pt")
        )


del dl_iter
writer.close()
