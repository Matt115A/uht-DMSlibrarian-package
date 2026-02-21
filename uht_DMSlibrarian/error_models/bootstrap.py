from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from .base import ErrorModel, ErrorModelConfig, ErrorModelFitResult

STAGES = [
    "library_composition",
    "library_prep_pcr",
    "droplet_sorting",
    "dictionary_assignment",
    "sequencing_counting",
    "replicate_residual",
]


class BootstrapErrorModel(ErrorModel):
    name = "bootstrap"

    @staticmethod
    def calculate_bootstrap_ci(
        df: pd.DataFrame,
        fitness_cols: list[str],
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        random_seed: int | None = 42,
    ) -> Dict[str, np.ndarray]:
        n_variants = len(df)
        n_replicates = len(fitness_cols)
        rng = np.random.default_rng(random_seed)

        bootstrap_fitness = np.zeros((n_bootstrap, n_variants), dtype=float)
        for b in range(n_bootstrap):
            resampled_indices = rng.choice(n_replicates, size=n_replicates, replace=True)
            resampled_cols = [fitness_cols[i] for i in resampled_indices]
            bootstrap_fitness[b, :] = df[resampled_cols].mean(axis=1).to_numpy()

        alpha = 1.0 - ci_level
        lower_percentile = (alpha / 2.0) * 100.0
        upper_percentile = (1.0 - alpha / 2.0) * 100.0

        ci_lower = np.percentile(bootstrap_fitness, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_fitness, upper_percentile, axis=0)
        std = np.std(bootstrap_fitness, axis=0)

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std": std,
        }

    def fit_and_predict(
        self,
        df: pd.DataFrame,
        fitness_cols: list[str],
        input_pools: list[str],
        output_pools: list[str],
        config: ErrorModelConfig,
    ) -> ErrorModelFitResult:
        n_reps = len(fitness_cols)

        if n_reps < 2:
            std = np.zeros(len(df), dtype=float)
            out = df[fitness_cols[0]].to_numpy() if n_reps == 1 else np.zeros(len(df), dtype=float)
            ci_lower = out.copy()
            ci_upper = out.copy()
            msg = "Single replicate: CI collapsed to point estimate"
            status = "ok_single_replicate"
        else:
            ci = self.calculate_bootstrap_ci(
                df,
                fitness_cols,
                n_bootstrap=config.n_bootstraps,
                ci_level=config.ci_level,
                random_seed=config.random_seed,
            )
            ci_lower = ci["ci_lower"]
            ci_upper = ci["ci_upper"]
            std = ci["std"]
            msg = "Replicate bootstrap completed"
            status = "ok"

        variant_metrics = pd.DataFrame(
            {
                "fitness_sigma": std,
                "fitness_ci_lower": ci_lower,
                "fitness_ci_upper": ci_upper,
                "error_model_status": status,
            },
            index=df.index,
        )
        var = np.clip(std**2, 1e-12, None)
        for s in STAGES:
            variant_metrics[f"var_stage_{s}"] = var if s == "replicate_residual" else 0.0
            variant_metrics[f"frac_stage_{s}"] = 1.0 if s == "replicate_residual" else 0.0
        variant_metrics["dominant_error_stage"] = "replicate_residual"
        variant_metrics["stage_attribution_quality_flag"] = "bootstrap_no_stage_identifiability"
        variant_metrics["observed_var_reps"] = np.nan
        variant_metrics["predicted_var_reps"] = var

        return ErrorModelFitResult(
            model_name=self.name,
            status=status,
            message=msg,
            parameters={"n_bootstraps": float(config.n_bootstraps), "ci_level": float(config.ci_level)},
            parameter_cis={},
            diagnostics={
                "n_replicates": n_reps,
                "n_variants": len(df),
                "stage_global_summary": {
                    s: {
                        "mean_variance": float(np.mean(var)) if s == "replicate_residual" else 0.0,
                        "median_variance": float(np.median(var)) if s == "replicate_residual" else 0.0,
                        "sum_variance": float(np.sum(var)) if s == "replicate_residual" else 0.0,
                        "global_fraction": 1.0 if s == "replicate_residual" else 0.0,
                    }
                    for s in STAGES
                },
            },
            variant_metrics=variant_metrics,
        )
