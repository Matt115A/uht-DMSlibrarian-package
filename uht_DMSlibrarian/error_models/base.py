from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import pandas as pd


@dataclass
class ErrorModelConfig:
    model_name: str = "bootstrap"
    fit_error_model: bool = True
    n_bootstraps: int = 200
    ci_level: float = 0.95
    single_rep_mode: str = "fallback"  # fallback|fail|force
    random_seed: Optional[int] = 42
    params: Dict[str, Any] = field(default_factory=dict)
    report_html: bool = True
    report_html_path: Optional[str] = None
    stage_metrics_json: Optional[str] = None
    attribution_scale: str = "both"  # variance|fraction|both
    shrinkage: str = "none"  # none|normal_prior
    shrinkage_prior_mean: float = 0.0
    shrinkage_prior_var: float = 1.0
    droplet_model_mode: str = "mechanistic_v1"


@dataclass
class ErrorModelFitResult:
    model_name: str
    status: str
    message: str
    parameters: Dict[str, float]
    parameter_cis: Dict[str, Dict[str, float]]
    diagnostics: Dict[str, Any]
    variant_metrics: pd.DataFrame


class ErrorModel:
    name = "base"

    def fit_and_predict(
        self,
        df: pd.DataFrame,
        fitness_cols: list[str],
        input_pools: list[str],
        output_pools: list[str],
        config: ErrorModelConfig,
    ) -> ErrorModelFitResult:
        raise NotImplementedError
