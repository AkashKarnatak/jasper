"""
Precompute T5 prompt embeddings for all LIBERO tasks.

Saves a single .pt file mapping task_name -> prompt_embedding tensor.

Usage:
    python -m jasper.libero.precompute_prompts \
        --dataset-dir /home/ubuntu/workspace/LIBERO/libero/datasets/libero_90 \
        --output-path /home/ubuntu/workspace/LIBERO/libero/datasets/libero_90/prompt_embeds.pt
"""

import argparse
import os
import torch
from glob import glob
from diffusers import Cosmos2VideoToWorldPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(args.dataset_dir, "prompt_embeds.pt")

    # Load tokenizer and text encoder from Cosmos pipeline
    print("Loading Cosmos pipeline for T5 text encoder...")
    pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
        "nvidia/Cosmos-Predict2-2B-Video2World",
    )
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    text_encoder.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder = text_encoder.to(device)

    # Collect unique task names
    hdf5_paths = sorted(glob(os.path.join(args.dataset_dir, "*.hdf5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"No HDF5 files found in {args.dataset_dir}")

    task_names = []
    for hdf5_path in hdf5_paths:
        name = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
        name = ' '.join(name.split('SCENE')[-1].split('_')[1:])
        task_names.append(name)

    print(f"Found {len(task_names)} tasks")

    # Compute embeddings
    prompt_embeds = {}
    with torch.no_grad():
        for i, task_name in enumerate(task_names):
            text_inputs = tokenizer(
                [task_name],
                padding="max_length",
                max_length=args.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.bool().to(device)

            embeds = text_encoder(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state  # (1, seq_len, dim)

            # Zero out padding positions
            length = attention_mask.sum().item()
            embeds[0, length:] = 0

            prompt_embeds[task_name] = embeds.squeeze(0).cpu()  # (seq_len, dim)
            print(f"  [{i+1}/{len(task_names)}] {task_name} -> {embeds.shape}")

    torch.save(prompt_embeds, args.output_path)
    print(f"\nSaved {len(prompt_embeds)} prompt embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
