"""Inspect the saved checkpoint for training metadata (loss, epoch, metrics)."""
import torch
import sys

weights_path = r"d:\Project 2\ai-service\weights\hybrid_detector_best.pt"

print(f"Loading checkpoint: {weights_path}")
checkpoint = torch.load(weights_path, map_location="cpu")

if isinstance(checkpoint, dict):
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    for key in checkpoint:
        if key == "model_state":
            print(f"  {key}: <state_dict with {len(checkpoint[key])} parameters>")
        elif key == "config":
            print(f"  {key}: {checkpoint[key]}")
        else:
            val = checkpoint[key]
            if isinstance(val, (int, float, str, bool, list, tuple)):
                print(f"  {key}: {val}")
            elif isinstance(val, dict) and len(str(val)) < 2000:
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: <{type(val).__name__}, len={len(val) if hasattr(val, '__len__') else 'N/A'}>")
else:
    print(f"Checkpoint is a raw state_dict (type: {type(checkpoint).__name__})")
    print(f"  Number of parameters: {len(checkpoint)}")
    print("  No training metadata found in checkpoint.")
