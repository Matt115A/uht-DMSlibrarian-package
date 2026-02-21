from pathlib import Path

from uht_DMSlibrarian.error_models.config import merge_error_model_config


def test_cli_overrides_defaults():
    cfg = merge_error_model_config(
        cli_model="droplet_full",
        cli_n_bootstraps=77,
        cli_single_rep_mode="force",
        cli_report_html=True,
    )
    assert cfg.model_name == "droplet_full"
    assert cfg.n_bootstraps == 77
    assert cfg.single_rep_mode == "force"
    assert cfg.report_html is True


def test_json_config_loading(tmp_path: Path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"model":"dimsum_analog","n_bootstraps":12,"single_rep_mode":"fallback"}')

    cfg = merge_error_model_config(cli_config_path=str(cfg_path))
    assert cfg.model_name == "dimsum_analog"
    assert cfg.n_bootstraps == 12
    assert cfg.single_rep_mode == "fallback"


def test_shrinkage_config_override():
    cfg = merge_error_model_config(
        cli_model="droplet_full",
        cli_shrinkage="normal_prior",
        cli_shrinkage_prior_mean=0.15,
        cli_shrinkage_prior_var=0.4,
    )
    assert cfg.model_name == "droplet_full"
    assert cfg.shrinkage == "normal_prior"
    assert cfg.shrinkage_prior_mean == 0.15
    assert cfg.shrinkage_prior_var == 0.4
