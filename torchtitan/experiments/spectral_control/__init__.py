# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from torchtitan.experiments.qwen3.infra.parallelize import parallelize_qwen3
from torchtitan.experiments.qwen3.__init__ import qwen3_configs
from torchtitan.experiments.qwen3.model.model import Qwen3Model

from .projector import attach_spectral_projection

__all__ = [
    "parallelize_qwen3",
    "Qwen3Model",
    "spectral_qwen3_configs",
]


# Reuse Qwen3 configs; users can select the same flavors (start with 0.6B)
spectral_qwen3_configs = qwen3_configs


def build_optimizers_with_spectral(*args, **kwargs):
    optimizers = build_optimizers(*args, **kwargs)
    # Attach spectral projection as an optimizer post-step hook
    attach_spectral_projection(optimizers)
    return optimizers


register_train_spec(
    TrainSpec(
        name="spectral_control",
        model_cls=Qwen3Model,
        model_args=spectral_qwen3_configs,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers_with_spectral,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=None,
    )
)

