"""Common utilities."""

import argparse
import dataclasses
import json
import logging
import uuid

import torch


def get_run_name(prefix: str | None) -> str | None:
    """Given an optional prefix, returns a name with a unique suffix"""
    if prefix:
        return prefix + f"-{uuid.uuid4().hex[:6]}"
    return None


def save_argparse(args: argparse.Namespace, out_path: str) -> None:
    """Serializes the argparse.Namespace to a JSON file.

    Args:
        args: The parsed command-line arguments.
        out_path: The path to save the JSON file.
    """
    config_dict = vars(args)
    with open(out_path, "w") as f:
        json.dump(config_dict, f)


def save_dataclass(obj: object, out_path: str) -> None:
    """Writes the dictionary representation of `dataclass` to disk."""
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f"Object of type {type(obj)} is not a dataclass.")
    obj_dict = dataclasses.asdict(obj)
    with open(out_path, "w") as f:
        json.dump(obj_dict, f)


def get_device() -> torch.device:
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_str = "mps"
    return torch.device(device_str)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        logging.info("Synchronizing CUDA backend...")
        torch.cuda.synchronize()
    elif device.type == "mps":
        logging.info("Synchronizing MPS backend...")
        torch.mps.synchronize()
    elif device.type == "cpu":
        logging.info("Execution on CPU, no synchornization required.")
    else:
        raise ValueError(f"Unknown device type: {device.type}.")


def get_dtype(dtype_str: str) -> torch.dtype:
    """Returns a torch dtype for a corresponding string (e.g. 'float16')."""
    return getattr(torch, dtype_str.lower())
