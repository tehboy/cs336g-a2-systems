"""Leaderboard timing for CS336g FlashAttention2 Triton kernel.

Usage:
```python
uv run python cs336_systems/triton_leaderboard.py \
  --warmup=1000 \
  --rep=10000 \
  --n_heads=16 \
  --d_head=64 \
  --sequence_length=16384 \
  --dtype="float16" \
  --log_level="info"
```
"""

import argparse
import logging
import statistics
import sys
from collections.abc import Sequence

import torch
import triton
from jaxtyping import Float

from cs336_systems import flash_attention
from cs336_systems import utils


def get_tensors(
    n_heads: int,
    context_length: int,
    d_head: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Float[torch.Tensor, "nh t d"], ...]:
    """Returns a tuple of random tensors q,k,v for benchmarking."""
    return torch.randn(
        3,
        n_heads,
        context_length,
        d_head,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parses command-line arguments for the leaderboard."""
    parser = argparse.ArgumentParser(
        description="Leaderboard benchmark for FlashAttention2."
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="The number of samples to take when computing the average timing.",
        required=False,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1_000,
        help="The warmup time (given in ms).",
        required=False,
    )

    parser.add_argument(
        "--rep",
        type=int,
        default=10_000,
        help="The repetition time (given in ms).",
        required=False,
    )

    parser.add_argument(
        "--n_heads",
        type=int,
        default=16,
        help="The number of attention heads.",
        required=False,
    )

    parser.add_argument(
        "--d_head",
        type=int,
        default=64,
        help="The attention head dimension.",
        required=False,
    )

    parser.add_argument(
        "--sequence_length",
        type=int,
        default=16384,
        help="The number of tokens in the sequence.",
        required=False,
    )

    parser.add_argument(
        "--dtype",
        default="float16",
        help="The dtype of the Q/K/V tensors to use.",
        required=False,
        type=str,
        choices=[
            "bfloat16",
            "float16",
            "float32",
        ],
    )

    parser.add_argument(
        "--log_level",
        default="warning",
        help="Set the logging level for the binary.",
        required=False,
        type=str,
        choices=[
            "critical",
            "error",
            "warning",
            "info",
            "debug",
        ],
    )

    return parser.parse_args(argv[1:])


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper())

    device = utils.get_device()
    logging.info("Using device: %s...", device)
    if "cuda" not in str(device):
        raise ValueError("Triton benchmark only supports CUDA devices.")

    dtype = utils.get_dtype(args.dtype)
    logging.info("Using dtype: %s...", dtype)

    q, k, v = get_tensors(
        args.n_heads,
        args.sequence_length,
        args.d_head,
        dtype,
        device,
    )

    logging.info("Compiling FlashAttentionFunc...")
    flash = torch.compile(flash_attention.FlashAttentionFunc.apply)

    def flash_fwd_bwd() -> None:
        o = flash(q, k, v, True)
        loss = o.sum()
        _ = loss.backward()

    logging.info("Running benchmark...")

    results = []
    for i in range(args.num_samples):
        logging.info("Taking sample %d...", i)
        avg_ms = triton.testing.do_bench(
            flash_fwd_bwd,
            rep=args.rep,
            warmup=args.warmup,
        )
        results.append(avg_ms)

    print(f"Average execution time (ms): {statistics.mean(results)}")


if __name__ == "__main__":
    main(sys.argv)
