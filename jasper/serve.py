"""
Usage:
    python -m jasper.serve --ckpt-path ./ckpts/libero/checkpoint_24000.pt --compile
    python -m jasper.serve --ckpt-path ./ckpts/robotwin/checkpoint_10000.pt

Protocol:
    Client sends msgpack with:
        {"views": [[frame, frame, ...], [frame, frame, ...], ...]}
    where each view is a list of T serialized frames (dict with "data", "shape", "dtype"),
    and each frame is a (H, W, 3) uint8 numpy array.

    Server responds with:
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


def decode_view(frames_list):
    """Decode a list of serialized frames for one view and preprocess with V-JEPA2.

    Args:
        frames_list: list of T dicts with keys "data", "shape", "dtype"

    Returns:
        Tensor of shape (3, T, 256, 256)
    """
    frames = []
    for frame_msg in frames_list:
        arr = np.frombuffer(frame_msg["data"], dtype=frame_msg["dtype"]).reshape(
            frame_msg["shape"]
        )
        frames.append(arr)
    return processor(frames)[0]  # (3, T, 256, 256)


def preprocess_views(views):
    """Preprocess N views into (1, N, 3, T, 256, 256).

    Args:
        views: list of N lists, each containing T serialized frames.

    Returns:
        Tensor of shape (1, N, 3, T, 256, 256)
    """
    tensors = [decode_view(view) for view in views]
    return torch.stack(tensors).unsqueeze(0).to(config.device)  # (1, N, 3, T, 256, 256)


async def handle_client(websocket):
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for raw_message in websocket:
            msg = msgpack.unpackb(raw_message, raw=False)
            views = msg["views"]  # list of N views, each with T frames

            for i, view in enumerate(views):
                if len(view) != CHUNK_SIZE:
                    error = msgpack.packb(
                        {"error": f"View {i}: expected {CHUNK_SIZE} frames, got {len(view)}"}
                    )
                    await websocket.send(error)
                    break
            else:
                with torch.no_grad():
                    images = preprocess_views(views)
                    actions = model.sample_action(
                        images, num_steps=30
                    )  # (1, action_horizon, action_dim)

                actions_np = actions.cpu().numpy()
                response = msgpack.packb(
                    {
                        "actions": {
                            "data": actions_np.tobytes(),
                            "shape": list(actions_np.shape),
                            "dtype": str(actions_np.dtype),
                        },
                    }
                )
                await websocket.send(response)
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")


async def serve():
    host = "0.0.0.0"
    port = 8765
    print(f"Starting server on ws://{host}:{port}")
    async with websockets.serve(handle_client, host, port, max_size=100 * 1024 * 1024):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs the policy server"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--compile", action='store_true', help="Whether to compile vision encoder or not"
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    ckpt_dir = ckpt_path.parent

    config = JasperConfig(**load_config(ckpt_dir))
    model = Jasper(config).to(config.device, non_blocking=True)
    processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")

    if args.compile:
        model.vision_encoder = torch.compile(model.vision_encoder)
    state_dict = torch.load(str(ckpt_path))
    model.load_state_dict(state_dict)
    model.eval()

    CHUNK_SIZE = config.action_horizon

    asyncio.run(serve())
