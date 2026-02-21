from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import ErrorModel, ErrorModelConfig, ErrorModelFitResult
from .bootstrap import BootstrapErrorModel


STAGES = [
    "library_composition",
    "library_prep_pcr",
    "droplet_sorting",
    "dictionary_assignment",
    "sequencing_counting",
    "replicate_residual",
]


def _safe_fraction(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numer, dtype=float)
    mask = denom > 0
    out[mask] = numer[mask] / denom[mask]
    return out


def _load_stage_metrics(config: ErrorModelConfig) -> Dict[str, Any]:
    path = config.stage_metrics_json or config.params.get("stage_metrics_json")
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        loaded = json.loads(p.read_text())
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _extract_stage_priors(stage_metrics: Dict[str, Any]) -> Dict[str, float]:
    match_rate = float(stage_metrics.get("global_match_rate", 0.95))
    mismatch_rate = float(np.clip(1.0 - match_rate, 0.0, 0.5))

    faulty_sorting_rate = float(stage_metrics.get("faulty_sorting_rate", stage_metrics.get("sorting_false_positive_rate", 0.02)))
    faulty_sorting_rate = float(np.clip(faulty_sorting_rate, 0.0, 0.25))

    droplet_mu = float(stage_metrics.get("droplet_occupancy_mu", 0.3))
    droplet_mu = max(droplet_mu, 0.0)
    occupancy_error = float(1.0 - np.exp(-droplet_mu) * (1.0 + droplet_mu + 0.5 * droplet_mu * droplet_mu))
    occupancy_error = float(np.clip(occupancy_error, 0.0, 0.25))

    gate_percentile = float(stage_metrics.get("sorting_target_percentile", 95.0))
    gate_percentile = float(np.clip(gate_percentile, 1.0, 99.9))
    gate_strength = float(np.clip((100.0 - gate_percentile) / 100.0, 0.001, 0.5))

    return {
        "mismatch_rate": mismatch_rate,
        "faulty_sorting_rate": faulty_sorting_rate,
        "occupancy_error": occupancy_error,
        "gate_strength": gate_strength,
    }


def _compose_mechanistic_stage_components(
    params: np.ndarray,
    counts_in: np.ndarray,
    counts_out: np.ndarray,
    n_reps: int,
    priors: Dict[str, float],
) -> Dict[str, np.ndarray]:
    a_in = params[:n_reps]
    a_out = params[n_reps : 2 * n_reps]
    rep = params[2 * n_reps : 3 * n_reps]
    (
        phi_lib,
        phi_prep,
        phi_sort,
        phi_seq,
        phi_dict,
        kappa_pcr,
        kappa_seq,
        gate_jitter,
        p_fp,
        p_fn,
        epsilon_dict,
        sigma_floor,
    ) = params[3 * n_reps :]

    mean_in = np.mean(counts_in, axis=1)
    mean_out = np.mean(counts_out, axis=1)

    in_term = np.sum(a_in[None, :] / (counts_in + 1.0), axis=1) / (n_reps**2)
    out_term = np.sum(a_out[None, :] / (counts_out + 1.0), axis=1) / (n_reps**2)
    rep_term = float(np.sum(rep) / (n_reps**2))

    depth_joint = np.sqrt(np.maximum(mean_in, 0.0) * np.maximum(mean_out, 0.0))
    enrich_proxy = mean_out / (mean_in + mean_out + 1.0)
    pass_prob = np.clip(enrich_proxy, 1e-6, 1.0 - 1e-6)
    gate_var = pass_prob * (1.0 - pass_prob)

    mismatch_rate = priors["mismatch_rate"]
    faulty_sorting_rate = priors["faulty_sorting_rate"]
    occupancy_error = priors["occupancy_error"]
    gate_strength = priors["gate_strength"]

    stage_library = 0.30 * in_term + phi_lib / (mean_in + 1.0)
    stage_prep = phi_prep / (mean_in + 1.0) + kappa_pcr / (np.sqrt(mean_in) + 1.0)
    stage_sort = (
        phi_sort / (mean_out + 1.0)
        + gate_jitter * gate_var * (1.0 + gate_strength)
        + (p_fp + faulty_sorting_rate) * pass_prob
        + (p_fn + occupancy_error) * (1.0 - pass_prob)
    )
    stage_dict = phi_dict / (mean_in + mean_out + 1.0) + epsilon_dict + mismatch_rate
    stage_seq = out_term + 0.70 * in_term + phi_seq / (depth_joint + 1.0) + kappa_seq / (mean_out + 1.0)
    stage_rep = np.full(len(mean_in), rep_term + sigma_floor)

    return {
        "library_composition": np.clip(stage_library, 1e-12, None),
        "library_prep_pcr": np.clip(stage_prep, 1e-12, None),
        "droplet_sorting": np.clip(stage_sort, 1e-12, None),
        "dictionary_assignment": np.clip(stage_dict, 1e-12, None),
        "sequencing_counting": np.clip(stage_seq, 1e-12, None),
        "replicate_residual": np.clip(stage_rep, 1e-12, None),
    }


def _compose_dimsum_stage_components(
    params: np.ndarray,
    counts_in: np.ndarray,
    counts_out: np.ndarray,
    n_reps: int,
) -> Dict[str, np.ndarray]:
    a_in = params[:n_reps]
    a_out = params[n_reps : 2 * n_reps]
    rep = params[2 * n_reps : 3 * n_reps]

    in_term = np.sum(a_in[None, :] / (counts_in + 1.0), axis=1) / (n_reps**2)
    out_term = np.sum(a_out[None, :] / (counts_out + 1.0), axis=1) / (n_reps**2)
    rep_term = float(np.sum(rep) / (n_reps**2))
    rep_vec = np.full(counts_in.shape[0], rep_term)

    stage_library = 0.5 * in_term
    stage_seq = out_term + 0.5 * in_term

    return {
        "library_composition": np.clip(stage_library, 1e-12, None),
        "library_prep_pcr": np.zeros_like(stage_library),
        "droplet_sorting": np.zeros_like(stage_library),
        "dictionary_assignment": np.zeros_like(stage_library),
        "sequencing_counting": np.clip(stage_seq, 1e-12, None),
        "replicate_residual": np.clip(rep_vec, 1e-12, None),
    }


def _fit_variance_model(
    counts_in: np.ndarray,
    counts_out: np.ndarray,
    observed_var: np.ndarray,
    n_reps: int,
    model_kind: str,
    stage_metrics: Dict[str, Any],
    seed: int | None = 42,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)

    priors = _extract_stage_priors(stage_metrics)

    if model_kind == "droplet_full":
        p0 = np.concatenate(
            [
                np.full(n_reps, 1.0),
                np.full(n_reps, 1.0),
                np.full(n_reps, 0.02),
                np.array([
                    0.5,
                    0.2,
                    0.5,
                    0.5,
                    0.15,
                    0.12,
                    0.20,
                    0.08,
                    max(priors["faulty_sorting_rate"] * 0.5, 1e-3),
                    max(priors["occupancy_error"] * 0.5, 1e-3),
                    max(priors["mismatch_rate"] * 0.5, 1e-3),
                    0.02,
                ]),
            ]
        )
    else:
        p0 = np.concatenate(
            [
                np.full(n_reps, 1.0),
                np.full(n_reps, 1.0),
                np.full(n_reps, 0.02),
            ]
        )

    p0 = p0 * (10 ** rng.normal(0.0, 0.1, size=len(p0)))

    mean_depth = np.mean(counts_in + counts_out, axis=1)

    def stage_components(params: np.ndarray) -> Dict[str, np.ndarray]:
        if model_kind == "droplet_full":
            return _compose_mechanistic_stage_components(
                params=params,
                counts_in=counts_in,
                counts_out=counts_out,
                n_reps=n_reps,
                priors=priors,
            )
        return _compose_dimsum_stage_components(params=params, counts_in=counts_in, counts_out=counts_out, n_reps=n_reps)

    def predict(params: np.ndarray) -> np.ndarray:
        comps = stage_components(params)
        pred = np.zeros(counts_in.shape[0], dtype=float)
        for stage in STAGES:
            pred += comps[stage]
        return pred

    def objective(params: np.ndarray) -> float:
        pred = np.clip(predict(params), 1e-10, None)
        obs = np.clip(observed_var, 1e-10, None)
        resid = np.log(obs) - np.log(pred)

        # Weight low-depth variants higher because they are most sensitive to mis-specified sampling terms.
        weights = 1.0 + 1.0 / (np.sqrt(np.maximum(mean_depth, 0.0)) + 1.0)
        weighted_mse = np.mean((resid * weights) ** 2)

        reg = 1e-4 * np.mean(params**2)
        if model_kind == "droplet_full":
            p_fp = params[3 * n_reps + 8]
            p_fn = params[3 * n_reps + 9]
            epsilon_dict = params[3 * n_reps + 10]
            soft_priors = (
                2.0 * (p_fp - priors["faulty_sorting_rate"]) ** 2
                + 2.0 * (p_fn - priors["occupancy_error"]) ** 2
                + 1.5 * (epsilon_dict - priors["mismatch_rate"]) ** 2
            )
            return float(weighted_mse + reg + soft_priors)
        return float(weighted_mse + reg)

    bounds = [(1e-6, 100.0)] * (2 * n_reps) + [(1e-8, 10.0)] * n_reps
    if model_kind == "droplet_full":
        bounds += [
            (1e-8, 50.0),  # phi_lib
            (1e-8, 50.0),  # phi_prep
            (1e-8, 50.0),  # phi_sort
            (1e-8, 50.0),  # phi_seq
            (1e-8, 50.0),  # phi_dict
            (1e-8, 20.0),  # kappa_pcr
            (1e-8, 20.0),  # kappa_seq
            (1e-8, 5.0),   # gate_jitter
            (1e-8, 0.25),  # p_fp
            (1e-8, 0.25),  # p_fn
            (1e-8, 0.20),  # epsilon_dict
            (1e-8, 2.0),   # sigma_floor
        ]

    result = minimize(objective, p0, method="L-BFGS-B", bounds=bounds)
    comps = stage_components(result.x)

    at_lower = []
    for i, (v, b) in enumerate(zip(result.x, bounds)):
        if abs(v - b[0]) < 1e-10:
            at_lower.append(i)

    hessian_condition = None
    try:
        if hasattr(result.hess_inv, "todense"):
            dense = np.asarray(result.hess_inv.todense(), dtype=float)
            eigvals = np.linalg.eigvalsh(dense)
            positive = eigvals[eigvals > 1e-12]
            if positive.size > 0:
                hessian_condition = float(np.max(positive) / np.min(positive))
    except Exception:
        hessian_condition = None

    diag = {
        "success": bool(result.success),
        "message": str(result.message),
        "objective": float(result.fun),
        "n_lower_bound_params": len(at_lower),
        "lower_bound_param_indices": at_lower,
        "hessian_condition": hessian_condition,
        "stage_metrics_used": bool(stage_metrics),
        "stage_priors": priors,
    }

    return result.x, diag, comps


def _global_stage_summary(comps: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    total_by_variant = np.zeros_like(next(iter(comps.values())))
    for stage in STAGES:
        total_by_variant += comps[stage]
    global_total = float(np.sum(total_by_variant))

    out: Dict[str, Dict[str, float]] = {}
    for stage in STAGES:
        sv = np.asarray(comps[stage], dtype=float)
        out[stage] = {
            "mean_variance": float(np.mean(sv)),
            "median_variance": float(np.median(sv)),
            "sum_variance": float(np.sum(sv)),
            "global_fraction": float(np.sum(sv) / global_total) if global_total > 0 else 0.0,
        }
    return out


def _compute_stage_confidence(
    frac_matrix: np.ndarray,
    n_lower_bound_params: int,
    n_params: int,
) -> np.ndarray:
    max_frac = np.max(frac_matrix, axis=1)
    bound_health = 1.0 - min(float(n_lower_bound_params) / max(float(n_params), 1.0), 0.8)
    # Keep confidence mostly variant-driven, with a moderate global identifiability penalty.
    scaled = max_frac * (0.70 + 0.30 * bound_health)
    return np.clip(scaled, 0.0, 1.0)


def _apply_optional_shrinkage(
    fit_avg: np.ndarray,
    sigma: np.ndarray,
    config: ErrorModelConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shrink_mode = str(getattr(config, "shrinkage", "none")).lower()
    if shrink_mode != "normal_prior":
        return fit_avg.copy(), np.zeros_like(fit_avg), np.ones_like(fit_avg)

    mu0 = float(getattr(config, "shrinkage_prior_mean", 0.0))
    tau2 = max(float(getattr(config, "shrinkage_prior_var", 1.0)), 1e-8)

    sigma2 = np.clip(sigma**2, 1e-12, None)
    weight_obs = tau2 / (tau2 + sigma2)
    shrunk = weight_obs * fit_avg + (1.0 - weight_obs) * mu0
    delta = shrunk - fit_avg
    return shrunk, delta, weight_obs


def _compose_variant_metrics(
    df: pd.DataFrame,
    fit: np.ndarray,
    comps: Dict[str, np.ndarray],
    status: str,
    ci_level: float,
    config: ErrorModelConfig,
    n_lower_bound_params: int,
    n_params: int,
) -> pd.DataFrame:
    var_total = np.clip(fit, 1e-12, None)
    sigma = np.sqrt(var_total)
    z = 1.96 if abs(ci_level - 0.95) < 1e-9 else 1.64
    fit_avg = df["fitness_avg"].to_numpy(dtype=float)

    shrunk, shrink_delta, shrink_factor = _apply_optional_shrinkage(fit_avg=fit_avg, sigma=sigma, config=config)

    data: Dict[str, Any] = {
        "fitness_sigma": sigma,
        "fitness_ci_lower": fit_avg - z * sigma,
        "fitness_ci_upper": fit_avg + z * sigma,
        "error_model_status": status,
        "fitness_shrunk": shrunk,
        "fitness_shrinkage_delta": shrink_delta,
        "fitness_shrinkage_factor": shrink_factor,
    }

    frac_dict: Dict[str, np.ndarray] = {}
    for stage in STAGES:
        frac = _safe_fraction(comps[stage], var_total)
        frac_dict[stage] = frac
        if config.attribution_scale in {"variance", "both"}:
            data[f"var_stage_{stage}"] = comps[stage]
        if config.attribution_scale in {"fraction", "both"}:
            data[f"frac_stage_{stage}"] = frac

    frac_matrix = np.vstack([frac_dict[stage] for stage in STAGES]).T
    max_idx = np.argmax(frac_matrix, axis=1)
    stage_conf = _compute_stage_confidence(frac_matrix, n_lower_bound_params=n_lower_bound_params, n_params=n_params)
    data["dominant_error_stage"] = [STAGES[i] for i in max_idx]
    data["stage_confidence_score"] = stage_conf
    data["stage_attribution_quality_flag"] = np.where(stage_conf < 0.35, "diffuse", "ok")

    return pd.DataFrame(data, index=df.index)


def _single_rep_fallback(
    model_name: str,
    message: str,
    df: pd.DataFrame,
    fitness_cols: list[str],
    input_pools: list[str],
    output_pools: list[str],
    config: ErrorModelConfig,
    min_sigma: float,
    flag: str,
) -> ErrorModelFitResult:
    fallback = BootstrapErrorModel().fit_and_predict(df, fitness_cols, input_pools, output_pools, config)
    fallback.model_name = model_name
    fallback.status = "fallback_single_replicate"
    fallback.message = message

    var = np.clip(np.maximum(fallback.variant_metrics["fitness_sigma"].to_numpy(), min_sigma) ** 2, 1e-12, None)
    fit = df["fitness_avg"].to_numpy(dtype=float)
    sigma = np.sqrt(var)

    fallback.variant_metrics["fitness_sigma"] = sigma
    fallback.variant_metrics["fitness_ci_lower"] = fit - 1.96 * sigma
    fallback.variant_metrics["fitness_ci_upper"] = fit + 1.96 * sigma

    shrunk, shrink_delta, shrink_factor = _apply_optional_shrinkage(fit_avg=fit, sigma=sigma, config=config)
    fallback.variant_metrics["fitness_shrunk"] = shrunk
    fallback.variant_metrics["fitness_shrinkage_delta"] = shrink_delta
    fallback.variant_metrics["fitness_shrinkage_factor"] = shrink_factor
    fallback.variant_metrics["stage_confidence_score"] = 0.0

    for stage in STAGES:
        if config.attribution_scale in {"variance", "both"}:
            fallback.variant_metrics[f"var_stage_{stage}"] = var if stage == "replicate_residual" else 0.0
        if config.attribution_scale in {"fraction", "both"}:
            fallback.variant_metrics[f"frac_stage_{stage}"] = 1.0 if stage == "replicate_residual" else 0.0

    fallback.variant_metrics["dominant_error_stage"] = "replicate_residual"
    fallback.variant_metrics["stage_attribution_quality_flag"] = flag
    fallback.variant_metrics["observed_var_reps"] = np.nan
    fallback.variant_metrics["predicted_var_reps"] = var

    fallback.diagnostics["low_replicate_confidence"] = True
    fallback.diagnostics["stage_global_summary"] = {
        stage: {
            "mean_variance": float(np.mean(var)) if stage == "replicate_residual" else 0.0,
            "median_variance": float(np.median(var)) if stage == "replicate_residual" else 0.0,
            "sum_variance": float(np.sum(var)) if stage == "replicate_residual" else 0.0,
            "global_fraction": 1.0 if stage == "replicate_residual" else 0.0,
        }
        for stage in STAGES
    }

    return fallback


class DiMSumAnalogErrorModel(ErrorModel):
    name = "dimsum_analog"

    def fit_and_predict(
        self,
        df: pd.DataFrame,
        fitness_cols: list[str],
        input_pools: list[str],
        output_pools: list[str],
        config: ErrorModelConfig,
    ) -> ErrorModelFitResult:
        n_reps = len(fitness_cols)
        stage_metrics = _load_stage_metrics(config)

        if n_reps < 2:
            if config.single_rep_mode == "fail":
                raise ValueError("dimsum_analog requires >=2 replicate pairs unless single_rep_mode != fail")
            return _single_rep_fallback(
                model_name=self.name,
                message="Single replicate fallback to bootstrap-like point estimate",
                df=df,
                fitness_cols=fitness_cols,
                input_pools=input_pools,
                output_pools=output_pools,
                config=config,
                min_sigma=0.0,
                flag="bootstrap_no_stage_identifiability",
            )

        fit_matrix = df[fitness_cols].to_numpy(dtype=float)
        observed_var = np.nanvar(fit_matrix, axis=1)
        counts_in = df[input_pools].to_numpy(dtype=float)
        counts_out = df[output_pools].to_numpy(dtype=float)

        params, diag, comps = _fit_variance_model(
            counts_in=counts_in,
            counts_out=counts_out,
            observed_var=observed_var,
            n_reps=n_reps,
            model_kind="dimsum_analog",
            stage_metrics=stage_metrics,
            seed=config.random_seed,
        )

        pred = np.zeros(len(df), dtype=float)
        for stage in STAGES:
            pred += comps[stage]

        status = "ok" if diag["success"] else "warning_fit"

        params_out: Dict[str, float] = {}
        a_in = params[:n_reps]
        a_out = params[n_reps : 2 * n_reps]
        rep = params[2 * n_reps : 3 * n_reps]
        for i in range(n_reps):
            params_out[f"a_in_{i+1}"] = float(a_in[i])
            params_out[f"a_out_{i+1}"] = float(a_out[i])
            params_out[f"sigma_rep_{i+1}"] = float(np.sqrt(max(rep[i], 0.0)))

        variant_metrics = _compose_variant_metrics(
            df=df,
            fit=pred,
            comps=comps,
            status=status,
            ci_level=config.ci_level,
            config=config,
            n_lower_bound_params=diag["n_lower_bound_params"],
            n_params=len(params),
        )
        variant_metrics["observed_var_reps"] = observed_var
        variant_metrics["predicted_var_reps"] = pred

        return ErrorModelFitResult(
            model_name=self.name,
            status=status,
            message=diag["message"],
            parameters=params_out,
            parameter_cis={},
            diagnostics={
                "n_replicates": n_reps,
                "n_variants": len(df),
                "objective": diag["objective"],
                "identifiability": {
                    "n_lower_bound_params": diag["n_lower_bound_params"],
                    "lower_bound_param_indices": diag["lower_bound_param_indices"],
                    "hessian_condition": diag["hessian_condition"],
                },
                "stage_global_summary": _global_stage_summary(comps),
                "stage_metrics_used": diag["stage_metrics_used"],
            },
            variant_metrics=variant_metrics,
        )


class DropletFullErrorModel(ErrorModel):
    name = "droplet_full"

    def fit_and_predict(
        self,
        df: pd.DataFrame,
        fitness_cols: list[str],
        input_pools: list[str],
        output_pools: list[str],
        config: ErrorModelConfig,
    ) -> ErrorModelFitResult:
        n_reps = len(fitness_cols)
        stage_metrics = _load_stage_metrics(config)

        if n_reps < 2 and config.single_rep_mode == "fail":
            raise ValueError("droplet_full requires >=2 replicate pairs unless single_rep_mode != fail")

        if n_reps < 2 and config.single_rep_mode in {"fallback", "force"}:
            return _single_rep_fallback(
                model_name=self.name,
                message="Single replicate fallback with prior inflation",
                df=df,
                fitness_cols=fitness_cols,
                input_pools=input_pools,
                output_pools=output_pools,
                config=config,
                min_sigma=0.2,
                flag="single_replicate_fallback",
            )

        fit_matrix = df[fitness_cols].to_numpy(dtype=float)
        observed_var = np.nanvar(fit_matrix, axis=1)
        counts_in = df[input_pools].to_numpy(dtype=float)
        counts_out = df[output_pools].to_numpy(dtype=float)

        params, diag, comps = _fit_variance_model(
            counts_in=counts_in,
            counts_out=counts_out,
            observed_var=observed_var,
            n_reps=n_reps,
            model_kind="droplet_full",
            stage_metrics=stage_metrics,
            seed=config.random_seed,
        )

        pred = np.zeros(len(df), dtype=float)
        for stage in STAGES:
            pred += comps[stage]

        status = "ok" if diag["success"] else "warning_fit"

        a_in = params[:n_reps]
        a_out = params[n_reps : 2 * n_reps]
        rep = params[2 * n_reps : 3 * n_reps]
        (
            phi_lib,
            phi_prep,
            phi_sort,
            phi_seq,
            phi_dict,
            kappa_pcr,
            kappa_seq,
            gate_jitter,
            p_fp,
            p_fn,
            epsilon_dict,
            sigma_floor,
        ) = params[3 * n_reps :]

        params_out: Dict[str, float] = {
            "droplet_model_mode": 1.0,
            "phi_lib": float(phi_lib),
            "phi_prep": float(phi_prep),
            "phi_sort": float(phi_sort),
            "phi_seq": float(phi_seq),
            "phi_dict": float(phi_dict),
            "kappa_pcr": float(kappa_pcr),
            "kappa_seq": float(kappa_seq),
            "gate_jitter": float(gate_jitter),
            "p_fp": float(p_fp),
            "p_fn": float(p_fn),
            "epsilon_dict": float(epsilon_dict),
            "sigma_floor": float(sigma_floor),
            "prior_faulty_sorting_rate": float(diag["stage_priors"]["faulty_sorting_rate"]),
            "prior_occupancy_error": float(diag["stage_priors"]["occupancy_error"]),
            "prior_mismatch_rate": float(diag["stage_priors"]["mismatch_rate"]),
        }
        for i in range(n_reps):
            params_out[f"a_in_{i+1}"] = float(a_in[i])
            params_out[f"a_out_{i+1}"] = float(a_out[i])
            params_out[f"sigma_rep_{i+1}"] = float(np.sqrt(max(rep[i], 0.0)))

        variant_metrics = _compose_variant_metrics(
            df=df,
            fit=pred,
            comps=comps,
            status=status,
            ci_level=config.ci_level,
            config=config,
            n_lower_bound_params=diag["n_lower_bound_params"],
            n_params=len(params),
        )
        variant_metrics["observed_var_reps"] = observed_var
        variant_metrics["predicted_var_reps"] = pred

        return ErrorModelFitResult(
            model_name=self.name,
            status=status,
            message=diag["message"],
            parameters=params_out,
            parameter_cis={},
            diagnostics={
                "n_replicates": n_reps,
                "n_variants": len(df),
                "objective": diag["objective"],
                "droplet_model_mode": config.droplet_model_mode,
                "identifiability": {
                    "n_lower_bound_params": diag["n_lower_bound_params"],
                    "lower_bound_param_indices": diag["lower_bound_param_indices"],
                    "hessian_condition": diag["hessian_condition"],
                },
                "stage_global_summary": _global_stage_summary(comps),
                "stage_metrics_used": diag["stage_metrics_used"],
                "stage_priors": diag["stage_priors"],
                "shrinkage": {
                    "mode": config.shrinkage,
                    "prior_mean": config.shrinkage_prior_mean,
                    "prior_var": config.shrinkage_prior_var,
                },
            },
            variant_metrics=variant_metrics,
        )
