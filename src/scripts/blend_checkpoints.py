import argparse
import os
from typing import Any, List

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata
import torch.serialization

torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, Any])


def _load_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _weighted_sum_tensor(a: torch.Tensor, b: torch.Tensor, wa: float, wb: float):
    if a.shape != b.shape:
        raise ValueError(f"Tensor shapes differ: {a.shape} vs {b.shape}")
    return wa * a + wb * b


def _merge_state_dicts(states: List[dict], weights: List[float]):
    keys = states[0].keys()
    merged = {}
    for key in keys:
        value = states[0][key]
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            accum = weights[0] * value
            for state, weight in zip(states[1:], weights[1:]):
                accum = accum + weight * state[key]
            merged[key] = accum
        else:
            merged[key] = value
    return merged


def blend_checkpoints(paths: List[str], weights: List[float], output_path: str):
    if len(paths) != len(weights):
        raise ValueError("Number of checkpoints and weights must match")
    if not paths:
        raise ValueError("No checkpoints provided")
    total = sum(weights)
    if total == 0:
        raise ValueError("Weights sum to zero")
    norm_weights = [w / total for w in weights]

    checkpoints = [_load_checkpoint(p) for p in paths]

    # Merge top-level tensors and metadata
    merged = {}
    for key in checkpoints[0]:
        if key == "state_dict":
            continue  # handled separately later
        first_val = checkpoints[0][key]
        if isinstance(first_val, torch.Tensor) and first_val.dtype.is_floating_point:
            accum = norm_weights[0] * first_val
            for ckpt, w in zip(checkpoints[1:], norm_weights[1:]):
                accum = accum + w * ckpt.get(key, torch.zeros_like(first_val))
            merged[key] = accum
        else:
            # Keep metadata from the highest-weight checkpoint (first after normalisation)
            merged[key] = first_val

    # Merge model parameters
    state_dicts = [ckpt["state_dict"] for ckpt in checkpoints]
    merged["state_dict"] = _merge_state_dicts(state_dicts, norm_weights)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(merged, output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Blend multiple PyTorch Lightning checkpoints.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of checkpoint paths to blend.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=False,
        help="Optional list of weights; defaults to equal averaging.",
    )
    parser.add_argument("--output", required=True, help="Path to save the blended checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = args.weights or [1.0] * len(args.inputs)
    blend_checkpoints(args.inputs, weights, args.output)
    print(f"Saved blended checkpoint to {args.output}")


if __name__ == "__main__":
    main()
