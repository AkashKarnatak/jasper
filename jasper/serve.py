"""
Usage:
    python -m jasper.serve --ckpt-path ./ckpts/libero/checkpoint_24000.pt --compile
    python -m jasper.serve --ckpt-path ./ckpts/libero/checkpoint_24000.pt --prompt-embeds ./prompt_embeds.pt

Protocol:
    1. Client sends handshake: {"type": "handshake"}
       Server responds: {"mode": "frames"|"latents", "chunk_size": N, "action_horizon": N}

    2a. Frame mode - client sends:
        {"type": "predict", "views": [[frame, ...], [frame, ...], ...]}
    2b. Latent mode - client sends:
        {"type": "predict", "latents": {"data": bytes, "shape": [...], "dtype": str},
         "prompt_embeds": {"data": bytes, "shape": [...], "dtype": str}}

    Server responds:
        {"actions": {"data": bytes, "shape": [1, T, action_dim], "dtype": "float32"}}
"""

import asyncio
import json
import msgpack
import numpy as np
import torch
import argparse
import websockets
from pathlib import Path
from .jasper import Jasper, JasperConfig


def load_config(ckpt_dir):
    with open(ckpt_dir / "config.json", "r") as f:
        config = json.load(f)
    return config


def deserialize_tensor(msg):
    """Deserialize a msgpack tensor dict to numpy array."""
    return np.frombuffer(msg["data"], dtype=msg["dtype"]).reshape(msg["shape"])


def decode_view(frames_list):
    frames = []
    for frame_msg in frames_list:
        arr = np.frombuffer(frame_msg["data"], dtype=frame_msg["dtype"]).reshape(
            frame_msg["shape"]
        )
        frames.append(arr)
    return processor(frames)[0]  # (3, T, 256, 256)


def preprocess_views(views):
    tensors = [decode_view(view) for view in views]
    return torch.stack(tensors).unsqueeze(0).to(config.device)


def predict_frames(msg):
    """Handle frame-based prediction (VJEPA mode)."""
    views = msg["views"]
    images = preprocess_views(views)
    actions = model.sample_action(images, num_steps=30)
    return actions


def predict_latents(msg):
    """Handle latent-based prediction (Cosmos mode)."""
    latents_np = deserialize_tensor(msg["latents"])
    latents = torch.from_numpy(latents_np.copy()).unsqueeze(0).to(config.device)

    prompt_np = deserialize_tensor(msg["prompt_embeds"])
    prompt_embs = torch.from_numpy(prompt_np.copy()).unsqueeze(0).to(config.device)

    actions = model.sample_action(latents, num_steps=30, prompt_embs=prompt_embs, t_v=0.42)
    return actions


def serialize_actions(actions):
    actions_np = actions.cpu().numpy()
    return {
        "actions": {
            "data": actions_np.tobytes(),
            "shape": list(actions_np.shape),
            "dtype": str(actions_np.dtype),
        },
    }


async def handle_client(websocket):
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for raw_message in websocket:
            msg = msgpack.unpackb(raw_message, raw=False)
            msg_type = msg.get("type", "predict")

            if msg_type == "handshake":
                response = msgpack.packb(
                    {
                        "mode": MODE,
                        "chunk_size": CHUNK_SIZE,
                        "action_horizon": config.action_horizon,
                    }
                )
                await websocket.send(response)
                continue

            with torch.no_grad():
                if MODE == "latents":
                    actions = predict_latents(msg)
                else:
                    actions = predict_frames(msg)

            await websocket.send(msgpack.packb(serialize_actions(actions)))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")


async def serve():
    host = "0.0.0.0"
    port = 8765
    print(f"Starting server on ws://{host}:{port}")
    print(f"Mode: {MODE}")
    async with websockets.serve(handle_client, host, port, max_size=100 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the policy server")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--prompt-embeds",
        type=str,
        default=None,
        help="Path to precomputed prompt embeddings (for Cosmos mode)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    ckpt_dir = ckpt_path.parent

    config = JasperConfig(**load_config(ckpt_dir))
    model = Jasper(config).to(config.device, non_blocking=True)

    # Determine mode from config
    if config.vjepa2_model is not None:
        MODE = "frames"
        processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")
    else:
        MODE = "latents"

    if args.compile:
        model.vision_encoder = torch.compile(model.vision_encoder)
    state_dict = torch.load(str(ckpt_path))
    model.load_state_dict(state_dict)
    model.eval()

    CHUNK_SIZE = config.action_horizon

    asyncio.run(serve())
