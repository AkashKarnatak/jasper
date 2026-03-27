import torch
import time
import json
from jasper import Jasper, JasperConfig
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from robotwin.dataset import RoboTwinDataset


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def dump_config(config, ckpt_dir):
    with open(str(ckpt_dir / "config.json"), "w") as f:
        json.dump(config.__dict__, f)


dataset_dir = "/home/ubuntu/workspace/RoboTwin/dataset"
ckpt_dir = Path("./ckpts/robotwin")
batch_size = 32
amp_dtype = torch.bfloat16
device = "cuda"
lr = (batch_size / 32) * 1e-4
weight_decay = 1e-3
max_grad_norm = 1.0
warmup_steps = 1000
save_every = 4000 // (batch_size // 32)
max_steps = 50_000 // (batch_size // 32)
writer = SummaryWriter(f"./logs/jasper-robotwin/{time.time()}")

ckpt_dir.mkdir(parents=True, exist_ok=True)

config = JasperConfig(
    device=device,
    dtype="float32",
    action_dim=14,
    action_horizon=30,
    hidden_dim=512,
    num_heads=8,
    head_dim=64,
    ff_dim=3200,
    attn_dropout=0.1,
    dropout=0.1,
    depth=4,
    vjepa2_model="vjepa2_1_vit_base_384",
)
model = Jasper(config).to(device, non_blocking=True)
torch.set_float32_matmul_precision("high")
scaler = torch.amp.GradScaler(enabled=False)
model.vision_encoder = torch.compile(model.vision_encoder, mode="max-autotune")

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

dataset = RoboTwinDataset(
    dataset_dir=dataset_dir,
    chunk_size=config.action_horizon,
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
dump_config(config, ckpt_dir)

for step in range(max_steps):
    batch = next(dl_iter)

    images = batch["cameras"]["head_camera"].to(device, non_blocking=True)
    action = batch["action"].to(device, non_blocking=True)

    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        loss = model(images, action)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    writer.add_scalar("Loss/Train", loss.item(), step)
    print(
        f"Step: {step:6d} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.2e}"
    )

    if (step + 1) % save_every == 0:
        torch.save(model.state_dict(), str(ckpt_dir / f"checkpoint_{step + 1}.pt"))


del dl_iter
writer.close()
