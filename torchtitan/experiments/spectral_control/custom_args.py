from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from torchtitan.config.job_config import JobConfig as BaseJobConfig


@dataclass
class Spectral:
    enable: bool = True
    method: Literal["soft_cap", "hard_cap", "normalize", "hammer", "weight_decay", "svd_cap", "none"] = "soft_cap"
    w_max: float = 1.0
    spectral_wd: float = 0.0
    alpha: float | None = None
    use_coupling: bool = True
    max_update_norm: float | None = None
    sensitive_to_wmax: bool = True
    include_fqns: list[str] = field(default_factory=list)
    exclude_fqns: list[str] = field(default_factory=lambda: ["tok_embeddings", "output"])


@dataclass
class JobConfig(BaseJobConfig):
    spectral: Spectral = field(default_factory=Spectral)

