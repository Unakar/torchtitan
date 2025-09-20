#!/usr/bin/env python
"""Static checks to verify Muon optimizer wiring without launching training."""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from torchtitan.config import ConfigManager
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims


def main() -> None:
    config_path = Path(__file__).resolve().parent / "muon_test.toml"

    cfg = ConfigManager().parse_args(["--job.config_file", str(config_path)])
    job = cfg.job
    parallel = cfg.parallelism

    print("Loaded config from", config_path)
    print("Optimizer name:", cfg.optimizer.name)
    print("Muon hyperparameters: mu=", cfg.optimizer.mu, " adjust_lr=", cfg.optimizer.adjust_lr)

    # Build a dummy ParallelDims that matches single-rank defaults for static validation
    world_size = parallel.data_parallel_shard_degree
    if world_size <= 0:
        world_size = 1
    parallel_dims = ParallelDims(
        dp_shard=parallel.data_parallel_shard_degree if parallel.data_parallel_shard_degree > 0 else 1,
        dp_replicate=parallel.data_parallel_replicate_degree,
        cp=parallel.context_parallel_degree,
        tp=parallel.tensor_parallel_degree,
        pp=parallel.pipeline_parallel_degree,
        ep=parallel.expert_parallel_degree,
        etp=parallel.expert_tensor_parallel_degree,
        world_size=world_size,
    )

    # Minimal model stub: reuse tokenizer config to avoid instantiating the real model.
    print("Building optimizers for a dummy module list (no parameters) just to ensure constructor runs...")
    dummy_model = nn.Module()

    try:
        optim_container = build_optimizers([dummy_model], cfg.optimizer, parallel_dims, None)
        print("Optimizer container type:", type(optim_container).__name__)
        print("Number of underlying optimizers:", len(list(optim_container)))
    except ValueError as e:
        if "RANK expected" in str(e):
            print("Skipping optimizer container build - requires distributed environment")
            print("This is expected when running static checks without distributed setup")
        else:
            raise e

    print("Config description:", job.description)


if __name__ == "__main__":
    main()
