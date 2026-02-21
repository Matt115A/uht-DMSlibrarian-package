from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ErrorModelConfig


DEFAULT_CONFIG = ErrorModelConfig()


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def load_error_model_config_file(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Error model config file not found: {config_path}")

    suffix = path.suffix.lower()
    raw = path.read_text()
    if suffix in {".json"}:
        return json.loads(raw)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML configs. Install with: pip install pyyaml") from exc
        return yaml.safe_load(raw) or {}

    # Attempt JSON fallback for unknown extensions
    try:
        return json.loads(raw)
    except Exception:
        raise ValueError(f"Unsupported config format for file: {config_path}")


def merge_error_model_config(
    cli_model: Optional[str] = None,
    cli_config_path: Optional[str] = None,
    cli_fit_error_model: Optional[bool] = None,
    cli_n_bootstraps: Optional[int] = None,
    cli_single_rep_mode: Optional[str] = None,
    cli_report_html: Optional[bool] = None,
    cli_report_html_path: Optional[str] = None,
    cli_random_seed: Optional[int] = None,
    cli_stage_metrics_json: Optional[str] = None,
    cli_attribution_scale: Optional[str] = None,
    cli_shrinkage: Optional[str] = None,
    cli_shrinkage_prior_mean: Optional[float] = None,
    cli_shrinkage_prior_var: Optional[float] = None,
) -> ErrorModelConfig:
    cfg_data = load_error_model_config_file(cli_config_path)

    model_name = cfg_data.get("model", cfg_data.get("error_model", DEFAULT_CONFIG.model_name))
    if cli_model is not None:
        model_name = cli_model

    fit_error_model = cfg_data.get("fit_error_model", DEFAULT_CONFIG.fit_error_model)
    if cli_fit_error_model is not None:
        fit_error_model = cli_fit_error_model

    n_bootstraps = int(cfg_data.get("n_bootstraps", cfg_data.get("error_model_bootstraps", DEFAULT_CONFIG.n_bootstraps)))
    if cli_n_bootstraps is not None:
        n_bootstraps = int(cli_n_bootstraps)

    single_rep_mode = cfg_data.get("single_rep_mode", DEFAULT_CONFIG.single_rep_mode)
    if cli_single_rep_mode is not None:
        single_rep_mode = cli_single_rep_mode

    report_block = cfg_data.get("report_html", {}) if isinstance(cfg_data.get("report_html", {}), dict) else {}
    report_html = _safe_bool(report_block.get("enabled", cfg_data.get("report_html_enabled", DEFAULT_CONFIG.report_html)), DEFAULT_CONFIG.report_html)
    if cli_report_html is not None:
        report_html = cli_report_html

    report_html_path = report_block.get("path", cfg_data.get("report_html_path", DEFAULT_CONFIG.report_html_path))
    if cli_report_html_path is not None:
        report_html_path = cli_report_html_path

    random_seed = cfg_data.get("random_seed", DEFAULT_CONFIG.random_seed)
    if cli_random_seed is not None:
        random_seed = cli_random_seed

    stage_metrics_json = cfg_data.get("stage_metrics_json", DEFAULT_CONFIG.stage_metrics_json)
    if cli_stage_metrics_json is not None:
        stage_metrics_json = cli_stage_metrics_json

    attribution_scale = cfg_data.get("attribution_scale", DEFAULT_CONFIG.attribution_scale)
    if cli_attribution_scale is not None:
        attribution_scale = cli_attribution_scale

    shrinkage = cfg_data.get("shrinkage", DEFAULT_CONFIG.shrinkage)
    if cli_shrinkage is not None:
        shrinkage = cli_shrinkage

    shrinkage_prior_mean = float(cfg_data.get("shrinkage_prior_mean", DEFAULT_CONFIG.shrinkage_prior_mean))
    if cli_shrinkage_prior_mean is not None:
        shrinkage_prior_mean = float(cli_shrinkage_prior_mean)

    shrinkage_prior_var = float(cfg_data.get("shrinkage_prior_var", DEFAULT_CONFIG.shrinkage_prior_var))
    if cli_shrinkage_prior_var is not None:
        shrinkage_prior_var = float(cli_shrinkage_prior_var)
    shrinkage_prior_var = max(shrinkage_prior_var, 1e-8)

    droplet_model_mode = str(cfg_data.get("droplet_model_mode", DEFAULT_CONFIG.droplet_model_mode))

    params = cfg_data.get("params", {}) if isinstance(cfg_data.get("params", {}), dict) else {}
    if stage_metrics_json:
        params["stage_metrics_json"] = stage_metrics_json
    params["shrinkage"] = str(shrinkage)
    params["shrinkage_prior_mean"] = float(shrinkage_prior_mean)
    params["shrinkage_prior_var"] = float(shrinkage_prior_var)
    params["droplet_model_mode"] = droplet_model_mode

    return ErrorModelConfig(
        model_name=str(model_name),
        fit_error_model=_safe_bool(fit_error_model, True),
        n_bootstraps=max(1, int(n_bootstraps)),
        ci_level=float(cfg_data.get("ci_level", DEFAULT_CONFIG.ci_level)),
        single_rep_mode=str(single_rep_mode),
        random_seed=int(random_seed) if random_seed is not None else None,
        params=params,
        report_html=_safe_bool(report_html, True),
        report_html_path=report_html_path,
        stage_metrics_json=stage_metrics_json,
        attribution_scale=str(attribution_scale),
        shrinkage=str(shrinkage),
        shrinkage_prior_mean=float(shrinkage_prior_mean),
        shrinkage_prior_var=float(shrinkage_prior_var),
        droplet_model_mode=droplet_model_mode,
    )
