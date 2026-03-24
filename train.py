import torch
from pathlib import Path
from itertools import cycle
from jasper import Jasper, JasperConfig
from dataset import RoboTwinDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


batch_size = 32
dtype = torch.float32
device = "cuda"
lr = (batch_size / 32) * 1e-5
weight_decay = 1e-4
config = JasperConfig(dtype=dtype, device=device)

model = Jasper(config).to(dtype).to(device, non_blocking=True)
torch.set_float32_matmul_precision("high")
# model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

ckpt_dir = Path("./ckpts")
num_steps = 10_000
save_every = 1000
# writer = SummaryWriter(f"./logs/act/{time.time()}")

ckpt_dir.mkdir(parents=True, exist_ok=True)

import multiprocessing
ctx = multiprocessing.get_context("spawn")
dataset = RoboTwinDataset("./dataset", split="train", chunk_size=30)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    multiprocessing_context=ctx,
)

model.train()
# for step in range(num_steps):
#     batch = next(dl_iter)
for step, batch in enumerate(dataloader):
    images = batch["cameras"]["head_camera"].to(dtype).to(device, non_blocking=True)
    action = batch["action"].to(dtype).to(device, non_blocking=True)

    loss = model(images, action)

    # writer.add_scalar("Loss/Train", loss.item(), step)
    print(f"Step: {step:6d} | Total Loss: {loss.item():.6f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if (step + 1) % save_every == 0:
    #     torch.save(
    #         model._orig_mod.state_dict(), str(ckpt_dir / f"checkpoint_{step + 1}.pt")
    #     )

del dl_iter
# writer.close()
