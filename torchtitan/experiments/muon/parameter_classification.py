# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parameter classification helpers for the Muon optimizer."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger


def _get_betas(config) -> Tuple[float, float]:
    betas = getattr(config, "betas", None)
    if isinstance(betas, (tuple, list)) and len(betas) == 2:
        return float(betas[0]), float(betas[1])
    beta1 = getattr(config, "beta1", 0.9)
    beta2 = getattr(config, "beta2", 0.95)
    return float(beta1), float(beta2)


def _get_epsilon(config) -> float:
    if hasattr(config, "epsilon"):
        return float(getattr(config, "epsilon"))
    return float(getattr(config, "eps", 1e-8))


def _get_mu(config) -> float:
    return float(getattr(config, "mu", 0.95))


def _get_matrix_algorithm(config) -> str:
    algorithm = getattr(config, "algorithm", None)
    if algorithm is None:
        return "muon"
    return str(algorithm).lower()


def _is_matrix_tensor(param: torch.Tensor, config) -> bool:
    if param.ndim == 2:
        return True
    if param.ndim > 2:
        return bool(getattr(config, "flatten", False))
    return False


def create_parameter_groups(
    model_parts: List[nn.Module],
    optimizer_config,
    *,
    log_summary: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, List]]:
    """Build Muon-aware optimizer parameter groups for the provided modules."""

    beta1, beta2 = _get_betas(optimizer_config)
    epsilon = _get_epsilon(optimizer_config)
    mu = _get_mu(optimizer_config)
    matrix_algorithm = _get_matrix_algorithm(optimizer_config)

    scalar_optimizer = getattr(optimizer_config, "scalar_optimizer", "adamw")
    embedding_optimizer = getattr(optimizer_config, "embedding_optimizer", "adamw")
    head_optimizer = getattr(optimizer_config, "head_optimizer", "adamw")
    routing_optimizer = getattr(optimizer_config, "routing_optimizer", "adamw")
    expert_optimizer = getattr(optimizer_config, "expert_optimizer", None)

    scalar_lr_factor = getattr(optimizer_config, "scalar_lr_factor", 1.0)
    embedding_lr_factor = getattr(optimizer_config, "embedding_lr_factor", 1.0)
    head_lr_factor = getattr(optimizer_config, "head_lr_factor", 1.0)
    routing_lr_factor = getattr(optimizer_config, "routing_lr_factor", 1.0)
    expert_lr_factor = getattr(optimizer_config, "expert_lr_factor", 1.0)
    head_lr_scaling = getattr(optimizer_config, "head_lr_scaling", True)

    param_groups: List[Dict[str, Any]] = []
    stats: Dict[str, List] = {
        "matrix": [],
        "scalar": [],
        "embedding": [],
        "head": [],
        "routing": [],
        "expert": [],
    }

    for model in model_parts:
        matrix_params: List[torch.Tensor] = []
        scalar_params: List[torch.Tensor] = []
        embedding_params: List[torch.Tensor] = []
        head_params: List[torch.Tensor] = []
        routing_params: List[torch.Tensor] = []
        expert_params: List[torch.Tensor] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param._param_name = name  # Helpful for downstream logging

            if is_expert_param(name, param, model):
                expert_type = classify_expert_param(name, param)
                stats["expert"].append((name, param.shape, expert_type))
                if expert_optimizer is not None:
                    expert_params.append(param)
                elif _is_matrix_tensor(param, optimizer_config) and not is_head_param(
                    name, param, model
                ):
                    matrix_params.append(param)
                else:
                    scalar_params.append(param)
                continue

            if is_routing_param(name, param, model):
                stats["routing"].append((name, param.shape))
                routing_params.append(param)
                continue

            if is_embedding_param(name, param, model):
                stats["embedding"].append((name, param.shape))
                embedding_params.append(param)
                continue

            if is_head_param(name, param, model):
                stats["head"].append((name, param.shape))
                head_params.append(param)
                continue

            if _is_matrix_tensor(param, optimizer_config):
                stats["matrix"].append((name, param.shape))
                matrix_params.append(param)
            else:
                stats["scalar"].append((name, param.shape))
                scalar_params.append(param)

        if matrix_params:
            param_groups.append(
                create_matrix_param_group(
                    matrix_params,
                    optimizer_config,
                    algorithm=matrix_algorithm,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    mu=mu,
                )
            )
        if scalar_params:
            param_groups.append(
                create_scalar_param_group(
                    scalar_params,
                    optimizer_config,
                    optimizer_name=scalar_optimizer,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    lr_factor=scalar_lr_factor,
                )
            )
        if embedding_params:
            param_groups.append(
                create_embedding_param_group(
                    embedding_params,
                    optimizer_config,
                    optimizer_name=embedding_optimizer,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    lr_factor=embedding_lr_factor,
                )
            )
        if head_params:
            param_groups.append(
                create_head_param_group(
                    head_params,
                    optimizer_config,
                    optimizer_name=head_optimizer,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    lr_factor=head_lr_factor,
                    apply_sqrt_scaling=head_lr_scaling,
                )
            )
        if routing_params:
            param_groups.append(
                create_routing_param_group(
                    routing_params,
                    optimizer_config,
                    optimizer_name=routing_optimizer,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    lr_factor=routing_lr_factor,
                )
            )
        if expert_params:
            param_groups.append(
                create_expert_param_group(
                    expert_params,
                    optimizer_config,
                    optimizer_name=expert_optimizer,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    lr_factor=expert_lr_factor,
                )
            )

    if log_summary:
        _log_parameter_summary(
            stats,
            scalar_optimizer,
            embedding_optimizer,
            head_optimizer,
            routing_optimizer,
            expert_optimizer,
            matrix_algorithm,
        )
    return param_groups, stats


def _log_parameter_summary(
    stats: Dict[str, List],
    scalar_optimizer: str,
    embedding_optimizer: str,
    head_optimizer: str,
    routing_optimizer: str,
    expert_optimizer: Optional[str],
    matrix_algorithm: str,
) -> None:
    logger.info("=" * 80)
    logger.info("PARAMETER OPTIMIZATION SUMMARY")
    logger.info("=" * 80)

    logger.info(
        f"Matrix parameters ({matrix_algorithm.upper()}): {len(stats['matrix'])}"
    )
    for name, shape in stats["matrix"]:
        logger.info(f"  - {name}: {shape}")

    logger.info(f"Scalar parameters ({scalar_optimizer.upper()}): {len(stats['scalar'])}")
    logger.info(
        f"Embedding parameters ({embedding_optimizer.upper()}): {len(stats['embedding'])}"
    )
    for name, shape in stats["embedding"]:
        logger.info(f"  - {name}: {shape}")

    logger.info(f"Head parameters ({head_optimizer.upper()}): {len(stats['head'])}")
    for name, shape in stats["head"]:
        logger.info(f"  - {name}: {shape}")

    logger.info(
        f"Routing parameters ({routing_optimizer.upper()}): {len(stats['routing'])}"
    )
    for name, shape in stats["routing"]:
        logger.info(f"  - {name}: {shape}")

    logger.info("=" * 40)
    logger.info("EXPERT WEIGHTS SUMMARY")
    logger.info("=" * 40)
    if not stats["expert"]:
        logger.info("No expert weight parameters detected in this model")
        logger.info("=" * 80)
        return

    if expert_optimizer is not None:
        logger.info(
            f"Expert optimizer configured: {expert_optimizer.upper()} (total {len(stats['expert'])})"
        )
        for name, shape, expert_type in stats["expert"]:
            logger.info(
                f"  ✓ EXPERT: {name} ({shape}) - {expert_type} → {expert_optimizer.upper()}"
            )
    else:
        logger.info(
            f"Expert optimizer not configured - defaulting to matrix/scalar classification ({len(stats['expert'])} parameters)"
        )
        for name, shape, expert_type in stats["expert"]:
            logger.info(f"  ✓ EXPERT: {name} ({shape}) - {expert_type}")
    logger.info("=" * 80)


def is_embedding_param(name: str, param: torch.Tensor, model: nn.Module) -> bool:
    embedding_patterns = [
        "embed",
        "embedding",
        "tok_embeddings",
        "word_embeddings",
        "position_embeddings",
        "pos_embed",
    ]
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in embedding_patterns)


def is_routing_param(name: str, param: torch.Tensor, model: nn.Module) -> bool:
    routing_patterns = [
        "router.gate",
        "gate.weight",
        "router_gate",
        "routing_gate",
        ".router.",
        "moe.router",
    ]
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in routing_patterns)


def is_head_param(name: str, param: torch.Tensor, model: nn.Module) -> bool:
    head_patterns = [
        "head",
        "output",
        "classifier",
        "lm_head",
        "prediction_head",
        "final_layer",
        "out_proj",
        "output_projection",
    ]
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in head_patterns)


def is_expert_param(name: str, param: torch.Tensor, model: nn.Module) -> bool:
    expert_patterns = [
        "experts.",
        ".expert.",
        "expert_",
        "moe.expert",
        "shared_experts",
        "routed_experts",
        ".experts[",
        ".w1.",
        ".w2.",
        ".w3.",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    name_lower = name.lower()
    for pattern in expert_patterns:
        if pattern in name_lower:
            if pattern in {"gate_proj", "up_proj", "down_proj"} and "expert" not in name_lower:
                continue
            return True
    return False


def classify_expert_param(name: str, param: torch.Tensor) -> str:
    name_lower = name.lower()
    if "shared_expert" in name_lower:
        return "Shared Expert"
    if "routed_expert" in name_lower:
        return "Routed Expert"
    if ".w1." in name_lower or "gate_proj" in name_lower:
        return "Expert Gate Projection"
    if ".w2." in name_lower or "down_proj" in name_lower:
        return "Expert Down Projection"
    if ".w3." in name_lower or "up_proj" in name_lower:
        return "Expert Up Projection"
    if "expert" in name_lower:
        return "Generic Expert"
    return "Unknown Expert Type"


def create_matrix_param_group(
    params: List[torch.Tensor],
    optimizer_config,
    *,
    algorithm: str,
    beta1: float,
    beta2: float,
    epsilon: float,
    mu: float,
) -> Dict[str, Any]:
    return {
        "params": params,
        "algorithm": algorithm,
        "lr": optimizer_config.lr,
        "mu": mu,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": optimizer_config.weight_decay,
        "epsilon": epsilon,
        "nesterov": getattr(optimizer_config, "nesterov", False),
        "adjust_lr": getattr(optimizer_config, "adjust_lr", None),
        "flatten": getattr(optimizer_config, "flatten", False),
    }


def create_scalar_param_group(
    params: List[torch.Tensor],
    optimizer_config,
    *,
    optimizer_name: str,
    beta1: float,
    beta2: float,
    epsilon: float,
    lr_factor: float,
) -> Dict[str, Any]:
    return {
        "params": params,
        "algorithm": optimizer_name,
        "lr": optimizer_config.lr * lr_factor,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": optimizer_config.weight_decay,
        "epsilon": epsilon,
    }


def create_embedding_param_group(
    params: List[torch.Tensor],
    optimizer_config,
    *,
    optimizer_name: str,
    beta1: float,
    beta2: float,
    epsilon: float,
    lr_factor: float,
) -> Dict[str, Any]:
    return {
        "params": params,
        "algorithm": optimizer_name,
        "lr": optimizer_config.lr * lr_factor,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": optimizer_config.weight_decay,
        "epsilon": epsilon,
    }


def _head_dim(param: torch.Tensor) -> int:
    if param.ndim >= 2:
        return int(param.shape[-1])
    return int(param.numel())


def create_head_param_group(
    params: List[torch.Tensor],
    optimizer_config,
    *,
    optimizer_name: str,
    beta1: float,
    beta2: float,
    epsilon: float,
    lr_factor: float,
    apply_sqrt_scaling: bool,
) -> Dict[str, Any]:
    lr = optimizer_config.lr * lr_factor
    if apply_sqrt_scaling and params:
        dim = _head_dim(params[0])
        if dim > 0:
            lr = lr / (dim**0.5)
    return {
        "params": params,
        "algorithm": optimizer_name,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": optimizer_config.weight_decay,
        "epsilon": epsilon,
    }


def create_routing_param_group(
    params: List[torch.Tensor],
    optimizer_config,
    *,
    optimizer_name: str,
    beta1: float,
    beta2: float,
    epsilon: float,
    lr_factor: float,
) -> Dict[str, Any]:
    return {
        "params": params,
        "algorithm": optimizer_name,
        "lr": optimizer_config.lr * lr_factor,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": optimizer_config.weight_decay,
        "epsilon": epsilon,
    }


def create_expert_param_group(
    params: List[torch.Tensor],
    optimizer_config,
    *,
    optimizer_name: str,
    beta1: float,
    beta2: float,
    epsilon: float,
    lr_factor: float,
) -> Dict[str, Any]:
    return {
        "params": params,
        "algorithm": optimizer_name,
        "lr": optimizer_config.lr * lr_factor,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": optimizer_config.weight_decay,
        "epsilon": epsilon,
    }
