# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DeviceMesh

from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims

from .muon import Muon
from .parameter_classification import create_parameter_groups

__all__ = [
    "MuonOptimizersContainer",
    "build_muon_optimizers",
]


class MuonOptimizersContainer(torch.optim.Optimizer, Stateful):
    """A container for Muon optimizers compatible with TorchTitan interface.

    This class wraps the Muon optimizer to make it compatible with the
    TorchTitan OptimizersContainer interface while preserving Muon's
    distributed training capabilities.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Configuration for Muon optimizer.
        parallel_dims (ParallelDims): Parallel dimensions configuration.
    """

    def __init__(
        self,
        model_parts: List[nn.Module],
        optimizer_config: OptimizerConfig,
        parallel_dims: ParallelDims,
    ) -> None:
        self.model_parts = model_parts
        self.optimizer_config = optimizer_config
        self.parallel_dims = parallel_dims

        # Setup device meshes from parallel dimensions
        distributed_mesh = self._setup_device_mesh(parallel_dims)

        self.optimizers: list[Muon] = []
        all_params: list[torch.nn.Parameter] = []

        for model in self.model_parts:
            param_groups, _ = create_parameter_groups(
                [model], optimizer_config, log_summary=False
            )
            if not param_groups:
                continue

            muon_optimizer = Muon(
                param_groups,
                distributed_mesh=distributed_mesh,
                lr=optimizer_config.lr,
                mu=optimizer_config.mu,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                weight_decay=optimizer_config.weight_decay,
                epsilon=optimizer_config.eps,
                nesterov=optimizer_config.nesterov,
                adjust_lr=optimizer_config.adjust_lr,
                flatten=optimizer_config.flatten,
                use_triton=optimizer_config.use_triton,
            )

            self.optimizers.append(muon_optimizer)
            for group in param_groups:
                for param in group["params"]:
                    if param.requires_grad:
                        all_params.append(param)

        # Emit a consolidated summary once after all optimizers are built.
        create_parameter_groups(self.model_parts, optimizer_config, log_summary=True)

        # Initialize torch Optimizer base to enable hooks and introspection utilities
        torch.optim.Optimizer.__init__(self, all_params, {"lr": optimizer_config.lr})

    def _setup_device_mesh(
        self, parallel_dims: ParallelDims
    ) -> Optional[Union[DeviceMesh, ProcessGroup]]:
        """Setup device mesh based on parallel dimensions.

        For Muon, we use the dp_shard mesh for distributed communication.
        """
        distributed_mesh = None

        # Get the world mesh from parallel_dims
        world_mesh = parallel_dims.world_mesh

        # For Muon, we primarily use the dp_shard mesh for distributed operations
        if parallel_dims.dp_shard_enabled:
            # Extract the dp_shard submesh
            if "dp_shard" in world_mesh.mesh_dim_names:
                distributed_mesh = world_mesh["dp_shard"]
            elif "dp_shard_cp" in world_mesh.mesh_dim_names:
                # If context parallel is enabled, use dp_shard_cp mesh
                distributed_mesh = world_mesh["dp_shard_cp"]
        elif parallel_dims.dp_replicate_enabled:
            # If no dp_shard but dp_replicate is enabled, use that
            if "dp_replicate" in world_mesh.mesh_dim_names:
                distributed_mesh = world_mesh["dp_replicate"]
            elif "dp" in world_mesh.mesh_dim_names:
                distributed_mesh = world_mesh["dp"]

        return distributed_mesh

    def __iter__(self):
        """Iterate over optimizers for compatibility."""
        return iter(self.optimizers)

    def __len__(self) -> int:
        """Return number of optimizers."""
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        """Perform optimization step."""
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Zero gradients for all optimizers."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict using distributed checkpoint utilities."""
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(
                func, self.model_parts, self.optimizers
            )
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict using distributed checkpoint utilities."""
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))


def build_muon_optimizers(
    model_parts: List[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
) -> MuonOptimizersContainer:
    """Create a MuonOptimizersContainer for the given model parts and config.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Muon optimizer configuration.
        parallel_dims (ParallelDims): Parallel dimensions for the model.

    Returns:
        MuonOptimizersContainer: Container with Muon optimizer.
    """
    return MuonOptimizersContainer(
        model_parts=model_parts,
        optimizer_config=optimizer_config,
        parallel_dims=parallel_dims,
    )
