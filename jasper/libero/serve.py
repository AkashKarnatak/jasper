import asyncio
import json
import msgpack
import numpy as np
import torch
import websockets
from pathlib import Path
from ..jasper import Jasper, JasperConfig


def load_config(ckpt_dir):
    with open(ckpt_dir / "config.json", "r") as f:
        config = json.load(f)
    return config


def preprocess_images(frames_bytes_list):
    """Decode a list of raw image arrays and preprocess with VJEPA2.

    Args:
        frames_bytes_list: list of 10 dicts with keys "data", "shape", "dtype"
                           each representing a (H, W, 3) uint8 numpy array.

    Returns:
        Tensor of shape (1, 3, T, 256, 256) ready for model input.
    """
    frames = []
    for frame_msg in frames_bytes_list:
        arr = np.frombuffer(frame_msg["data"], dtype=frame_msg["dtype"]).reshape(
            frame_msg["shape"]
        )
        frames.append(arr)

    # processor expects list of (H, W, 3) numpy arrays -> [(3, T, 256, 256)]
    processed = processor(frames)[0]  # (3, T, 256, 256)
    return processed.unsqueeze(0).to(config.device)  # (1, 3, T, 256, 256)


async def handle_client(websocket):
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for raw_message in websocket:
            msg = msgpack.unpackb(raw_message, raw=False)
            frames = msg["frames"]  # list of 10 serialized frames

            if len(frames) != CHUNK_SIZE:
                error = msgpack.packb(
                    {"error": f"Expected {CHUNK_SIZE} frames, got {len(frames)}"}
                )
                await websocket.send(error)
                continue

            with torch.no_grad():
                images = preprocess_images(frames)
                actions = model.sample_action(
                    images, num_steps=10
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
    ckpt_path = Path("./ckpts/libero/checkpoint_24000.pt")
    ckpt_dir = ckpt_path.parent

    config = JasperConfig(**load_config(ckpt_dir))
    model = Jasper(config).to(config.device, non_blocking=True)
    processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")

    model.vision_encoder = torch.compile(model.vision_encoder)
    state_dict = torch.load(str(ckpt_path))
    model.load_state_dict(state_dict)
    model.eval()

    CHUNK_SIZE = 10

    asyncio.run(serve())
