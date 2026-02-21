from __future__ import annotations

import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Mr Fish mascot – loaded once from package asset directory
# ---------------------------------------------------------------------------
_MR_FISH_B64: str | None = None
_ASSET_DIR = Path(__file__).parent


def _load_mr_fish() -> str:
    global _MR_FISH_B64
    if _MR_FISH_B64 is not None:
        return _MR_FISH_B64
    fish_path = _ASSET_DIR / "mr_fish.png"
    if fish_path.exists():
        _MR_FISH_B64 = base64.b64encode(fish_path.read_bytes()).decode("ascii")
    else:
        _MR_FISH_B64 = ""
    return _MR_FISH_B64


# ---------------------------------------------------------------------------
# Apple-inspired colour palette used across plots
# ---------------------------------------------------------------------------
_PALETTE = {
    "blue": "#007AFF",
    "indigo": "#5856D6",
    "purple": "#AF52DE",
    "teal": "#5AC8FA",
    "green": "#34C759",
    "orange": "#FF9500",
    "red": "#FF3B30",
    "pink": "#FF2D55",
    "gray": "#8E8E93",
    "text": "#1D1D1F",
    "secondary": "#86868B",
    "bg": "#F5F5F7",
    "card": "#FFFFFF",
}

_BAR_COLORS = [
    "#007AFF", "#5856D6", "#AF52DE", "#FF9500", "#34C759", "#FF3B30",
    "#5AC8FA", "#FF2D55", "#8E8E93", "#FFCC00",
]


def _apply_plot_style():
    """Configure matplotlib for Apple-like aesthetics."""
    matplotlib.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D2D2D7",
        "axes.labelcolor": "#1D1D1F",
        "text.color": "#1D1D1F",
        "xtick.color": "#86868B",
        "ytick.color": "#86868B",
        "grid.color": "#E5E5EA",
        "grid.alpha": 0.6,
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.titlepad": 14,
        "axes.labelsize": 11,
        "axes.labelpad": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 160,
    })


# ---------------------------------------------------------------------------
# CSS – Apple Human Interface Guidelines inspired
# ---------------------------------------------------------------------------
CSS = """
:root {
  --blue: #007AFF;
  --indigo: #5856D6;
  --teal: #5AC8FA;
  --green: #34C759;
  --orange: #FF9500;
  --red: #FF3B30;
  --text-primary: #1D1D1F;
  --text-secondary: #86868B;
  --bg: #F5F5F7;
  --card-bg: #FFFFFF;
  --border: #D2D2D7;
  --border-light: #E5E5EA;
  --shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.06);
  --shadow-hover: 0 2px 8px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.10);
  --radius: 16px;
  --radius-sm: 10px;
  --transition: 0.2s cubic-bezier(0.25, 0.1, 0.25, 1);
}

* { box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
    "Helvetica Neue", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--text-primary);
  line-height: 1.5;
}

.page-wrapper {
  max-width: 1080px;
  margin: 0 auto;
  padding: 40px 24px 80px;
}

/* ---- Header ---- */
.report-header {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 32px 36px;
  background: linear-gradient(135deg, #FFFFFF 0%, #F0F4FF 100%);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  margin-bottom: 12px;
}
.report-header img.mascot {
  width: 72px;
  height: 72px;
  border: none;
  border-radius: 16px;
  flex-shrink: 0;
}
.report-header .header-text h1 {
  margin: 0 0 2px;
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.5px;
  color: var(--text-primary);
}
.report-header .header-text .subtitle {
  font-size: 15px;
  color: var(--text-secondary);
  font-weight: 400;
}
.report-header .header-text .subtitle .brand {
  color: var(--blue);
  font-weight: 600;
}

.timestamp {
  text-align: right;
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 28px;
  padding-right: 4px;
}

/* ---- Navigation pills ---- */
.nav-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 32px;
}
.nav-pills a {
  display: inline-block;
  padding: 7px 16px;
  font-size: 13px;
  font-weight: 500;
  color: var(--blue);
  background: rgba(0, 122, 255, 0.08);
  border-radius: 980px;
  text-decoration: none;
  transition: background var(--transition), color var(--transition);
}
.nav-pills a:hover {
  background: var(--blue);
  color: #FFFFFF;
}

/* ---- Cards ---- */
.card {
  background: var(--card-bg);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 32px 36px;
  margin-bottom: 20px;
  transition: box-shadow var(--transition);
}
.card:hover {
  box-shadow: var(--shadow-hover);
}
.card h2 {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.3px;
  color: var(--text-primary);
  margin: 0 0 6px;
}
.card .section-desc {
  font-size: 14px;
  color: var(--text-secondary);
  margin: 0 0 24px;
  line-height: 1.55;
}
.card h3 {
  font-size: 17px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 28px 0 10px;
}

/* ---- Status badge ---- */
.badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 980px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.badge-ok { background: rgba(52,199,89,0.12); color: #248A3D; }
.badge-warn { background: rgba(255,149,0,0.12); color: #C93400; }
.badge-error { background: rgba(255,59,48,0.12); color: #D70015; }

/* ---- Key-value metadata ---- */
.kv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}
.kv-item {
  background: var(--bg);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
}
.kv-item .kv-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
  margin-bottom: 4px;
}
.kv-item .kv-value {
  font-size: 15px;
  font-weight: 500;
  color: var(--text-primary);
  word-break: break-word;
}

/* ---- Pre / Code ---- */
pre {
  background: var(--bg);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  padding: 16px 20px;
  font-family: "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 12.5px;
  line-height: 1.6;
  overflow-x: auto;
  color: var(--text-primary);
}
code {
  font-family: "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  background: rgba(0, 122, 255, 0.06);
  color: var(--blue);
  padding: 2px 6px;
  border-radius: 5px;
  font-size: 0.88em;
}

/* ---- Tables ---- */
table {
  border-collapse: separate;
  border-spacing: 0;
  width: 100%;
  font-size: 13px;
  border-radius: var(--radius-sm);
  overflow: hidden;
  border: 1px solid var(--border-light);
}
thead th {
  background: var(--bg);
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
  padding: 10px 14px;
  text-align: left;
  border-bottom: 1px solid var(--border-light);
}
tbody td {
  padding: 9px 14px;
  border-bottom: 1px solid var(--border-light);
  color: var(--text-primary);
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: rgba(0, 122, 255, 0.03); }
.table-wrapper { overflow-x: auto; border-radius: var(--radius-sm); }

/* ---- Images / plots ---- */
.plot-container {
  background: var(--card-bg);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  padding: 12px;
  margin: 16px 0;
  text-align: center;
}
.plot-container img {
  max-width: 100%;
  height: auto;
  border: none;
  border-radius: 8px;
}
img {
  max-width: 100%;
  border: none;
  border-radius: 8px;
}

/* ---- Collapsible details ---- */
details {
  margin: 16px 0;
}
details summary {
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  color: var(--blue);
  padding: 8px 0;
  user-select: none;
  list-style: none;
}
details summary::-webkit-details-marker { display: none; }
details summary::before {
  content: "+";
  display: inline-block;
  width: 22px;
  height: 22px;
  line-height: 22px;
  text-align: center;
  background: rgba(0,122,255,0.08);
  border-radius: 6px;
  margin-right: 10px;
  font-size: 14px;
  font-weight: 700;
  color: var(--blue);
  transition: transform var(--transition), background var(--transition);
}
details[open] summary::before {
  content: "-";
  background: rgba(0,122,255,0.14);
}
details .details-body {
  padding: 12px 0 0 32px;
}

/* ---- Artifacts list ---- */
.artifact-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.artifact-list li {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  background: var(--bg);
  border-radius: var(--radius-sm);
  margin-bottom: 8px;
  font-size: 13px;
}
.artifact-list li code {
  font-weight: 600;
}

/* ---- Utility ---- */
.mt-0 { margin-top: 0; }
.mb-0 { margin-bottom: 0; }
.text-secondary { color: var(--text-secondary); }
.text-sm { font-size: 13px; }

/* ---- Footer ---- */
.report-footer {
  text-align: center;
  padding: 32px 0 0;
  margin-top: 20px;
  border-top: 1px solid var(--border-light);
}
.report-footer img.mascot-sm {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 10px;
  opacity: 0.7;
  margin-bottom: 8px;
}
.report-footer p {
  font-size: 12px;
  color: var(--text-secondary);
  margin: 0;
}

/* ---- Identifiability list ---- */
.ident-list { list-style: none; padding: 0; }
.ident-list li {
  padding: 8px 16px;
  background: var(--bg);
  border-radius: var(--radius-sm);
  margin-bottom: 6px;
  font-size: 13px;
}

/* ---- Responsive ---- */
@media (max-width: 640px) {
  .page-wrapper { padding: 20px 12px 40px; }
  .card { padding: 20px 18px; border-radius: 12px; }
  .report-header { flex-direction: column; text-align: center; padding: 24px 20px; }
  .report-header img.mascot { width: 56px; height: 56px; }
  .report-header .header-text h1 { font-size: 22px; }
  .kv-grid { grid-template-columns: 1fr; }
}
"""


STAGE_DESCRIPTIONS = {
    "library_composition": "Input library sampling and composition imbalance effects.",
    "library_prep_pcr": "Amplification/library-prep variability before selection.",
    "droplet_sorting": "Sorting gate variability, false positive/negative sort events.",
    "dictionary_assignment": "Barcode/variant dictionary mismatch or assignment uncertainty.",
    "sequencing_counting": "Read-counting stochasticity from sequencing depth/noise.",
    "replicate_residual": "Residual replicate-to-replicate variance not explained by modeled stages.",
}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor="#FFFFFF", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _wrap_plot(b64: str) -> str:
    return f'<div class="plot-container"><img src="data:image/png;base64,{b64}"/></div>'


def _plot_uncertainty_vs_count(df: pd.DataFrame) -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.maximum(df.get("mean_input_count", pd.Series(np.ones(len(df)))).to_numpy(dtype=float), 1.0)
    y = df["fitness_sigma"].to_numpy(dtype=float)
    ax.scatter(np.log10(x), y, s=12, alpha=0.45, color=_PALETTE["blue"], edgecolors="none")
    ax.set_xlabel("log10 (mean input count)")
    ax.set_ylabel("Fitness sigma")
    ax.set_title("Uncertainty vs Input Count")
    return _fig_to_base64(fig)


def _plot_observed_vs_predicted(df: pd.DataFrame) -> str | None:
    if "observed_var_reps" not in df.columns or "predicted_var_reps" not in df.columns:
        return None
    x = np.clip(df["observed_var_reps"].to_numpy(dtype=float), 1e-10, None)
    y = np.clip(df["predicted_var_reps"].to_numpy(dtype=float), 1e-10, None)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return None

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(np.log10(x[valid]), np.log10(y[valid]), s=12, alpha=0.45,
               color=_PALETTE["indigo"], edgecolors="none")
    mn = min(np.min(np.log10(x[valid])), np.min(np.log10(y[valid])))
    mx = max(np.max(np.log10(x[valid])), np.max(np.log10(y[valid])))
    ax.plot([mn, mx], [mn, mx], color=_PALETTE["red"], ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("log10 (observed replicate variance)")
    ax.set_ylabel("log10 (predicted variance)")
    ax.set_title("Observed vs Predicted Variance")
    return _fig_to_base64(fig)


def _stage_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted([c for c in df.columns if c.startswith(prefix)])


def _plot_global_stage_fractions(stage_summary: Dict[str, Dict[str, float]]) -> str | None:
    if not stage_summary:
        return None
    stages = list(stage_summary.keys())
    frac = [float(stage_summary[s].get("global_fraction", 0.0)) for s in stages]
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = _BAR_COLORS[: len(stages)]
    bars = ax.bar(stages, frac, color=colors, width=0.6, edgecolor="none")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of modeled variance")
    ax.set_title("Global Error-Stage Attribution")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, frac):
        if v > 0.03:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=9, color=_PALETTE["secondary"])
    return _fig_to_base64(fig)


def _plot_dominant_stage_distribution(df: pd.DataFrame) -> str | None:
    if "dominant_error_stage" not in df.columns:
        return None
    counts = df["dominant_error_stage"].value_counts()
    if len(counts) == 0:
        return None
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = _BAR_COLORS[: len(counts)]
    ax.bar(counts.index.tolist(), counts.values, color=colors, width=0.6, edgecolor="none")
    ax.set_ylabel("Variant count")
    ax.set_title("Dominant Error Stage per Variant")
    ax.tick_params(axis="x", rotation=30)
    return _fig_to_base64(fig)


def _plot_stage_confidence(df: pd.DataFrame) -> str | None:
    if "stage_confidence_score" not in df.columns:
        return None
    vals = df["stage_confidence_score"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(vals, bins=25, color=_PALETTE["teal"], edgecolor="#FFFFFF", linewidth=0.6, alpha=0.85)
    ax.set_xlabel("Stage confidence score")
    ax.set_ylabel("Variants")
    ax.set_title("Stage Attribution Confidence")
    return _fig_to_base64(fig)


def _plot_shrinkage_delta(df: pd.DataFrame) -> str | None:
    if "fitness_shrinkage_delta" not in df.columns:
        return None
    vals = df["fitness_shrinkage_delta"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(vals, bins=30, color=_PALETTE["orange"], edgecolor="#FFFFFF", linewidth=0.6, alpha=0.85)
    ax.set_xlabel("Fitness shrinkage delta")
    ax.set_ylabel("Variants")
    ax.set_title("Shrinkage Impact Distribution")
    return _fig_to_base64(fig)


def _plot_model_flow(model_name: str) -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")
    stages = [
        "Input Counts",
        "Stage Variance\nDecomposition",
        "Total Sigma + CI",
        "Optional\nShrinkage",
        "Final Report",
    ]
    x_positions = np.linspace(0.08, 0.92, len(stages))
    for i, (x, label) in enumerate(zip(x_positions, stages)):
        ax.text(
            x, 0.50, label,
            ha="center", va="center", fontsize=10.5, fontweight="500",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#EFF4FF",
                  "edgecolor": "#007AFF", "linewidth": 0.8},
        )
        if i < len(stages) - 1:
            ax.annotate(
                "", xy=(x_positions[i + 1] - 0.07, 0.50),
                xytext=(x + 0.07, 0.50),
                arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#007AFF"},
            )
    ax.text(0.5, 0.92, f"Model Workflow  \u2014  {model_name}",
            ha="center", va="center", fontsize=12, fontweight="600", color=_PALETTE["text"])
    return _fig_to_base64(fig)


def _plot_stage_variance_composition() -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    ax.text(0.5, 0.95, "Stage Variance Composition (per variant)",
            ha="center", va="center", fontsize=13, color=_PALETTE["text"], weight="bold")

    ax.text(
        0.5, 0.84,
        "V_total  =  V_library  +  V_prep  +  V_sort  +  V_dict  +  V_seq  +  V_residual",
        ha="center", va="center", fontsize=11, fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#EFF4FF",
              "edgecolor": "#007AFF", "linewidth": 0.8},
    )

    stage_boxes = [
        ("V_library", "~ phi_lib / (mean_in + 1) + in_term"),
        ("V_prep", "~ phi_prep / (mean_in + 1) + kappa_pcr / (sqrt(mean_in) + 1)"),
        ("V_sort", "~ phi_sort / (mean_out + 1) + gate_jitter * gate_var + p_fp / p_fn terms"),
        ("V_dict", "~ phi_dict / (mean_in + mean_out + 1) + epsilon_dict + mismatch_rate"),
        ("V_seq", "~ out_term + in_term + phi_seq / (sqrt(mean_in * mean_out) + 1) + kappa_seq / (mean_out + 1)"),
        ("V_residual", "~ sigma_floor + replicate residual"),
    ]

    y0 = 0.70
    dy = 0.10
    for i, (name, formula) in enumerate(stage_boxes):
        y = y0 - i * dy
        color = _BAR_COLORS[i % len(_BAR_COLORS)]
        ax.text(
            0.07, y, name,
            ha="left", va="center", fontsize=10, fontweight="600",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": color + "18",
                  "edgecolor": color, "linewidth": 0.8},
        )
        ax.text(0.24, y, formula, ha="left", va="center", fontsize=9.5,
                fontfamily="monospace", color=_PALETTE["secondary"])

    ax.text(
        0.5, 0.06,
        "Predicted sigma  =  sqrt(V_total)       stage fractions  =  V_stage / V_total",
        ha="center", va="center", fontsize=10, color=_PALETTE["text"],
    )
    return _fig_to_base64(fig)


def _plot_shrinkage_workflow() -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")

    ax.text(0.5, 0.92, "Shrinkage Maths (per variant)",
            ha="center", va="center", fontsize=13, color=_PALETTE["text"], weight="bold")

    nodes = [
        ("Observed estimate", "fitness_avg\nvariance sigma\u00B2"),
        ("Prior", "\u03B8 ~ N(\u03BC\u2080, \u03C4\u00B2)"),
        ("Posterior weight", "w = \u03C4\u00B2 / (\u03C4\u00B2 + \u03C3\u00B2)"),
        ("Shrunk estimate", "fitness_shrunk =\nw\u00B7fitness_avg + (1\u2212w)\u00B7\u03BC\u2080"),
    ]
    xs = [0.12, 0.37, 0.62, 0.87]
    for (title, desc), x in zip(nodes, xs):
        ax.text(
            x, 0.52, f"{title}\n{desc}",
            ha="center", va="center", fontsize=10, fontweight="500",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#EFF4FF",
                  "edgecolor": "#007AFF", "linewidth": 0.8},
        )
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.09, 0.52), xytext=(xs[i] + 0.09, 0.52),
                     arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#007AFF"})

    ax.text(
        0.5, 0.12,
        "Large \u03C3\u00B2  \u2192  smaller w  \u2192  stronger pull toward prior \u03BC\u2080        "
        "Small \u03C3\u00B2  \u2192  w \u2248 1  \u2192  minimal shrinkage",
        ha="center", va="center", fontsize=9.5, color=_PALETTE["secondary"],
    )
    return _fig_to_base64(fig)


def _plot_stage_top_variants(df: pd.DataFrame) -> str | None:
    frac_cols = sorted([c for c in df.columns if c.startswith("frac_stage_")])
    if not frac_cols or "fitness_sigma" not in df.columns:
        return None
    top = df.sort_values("fitness_sigma", ascending=False).head(12).copy()
    if len(top) == 0:
        return None

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(top))
    bottom = np.zeros(len(top), dtype=float)
    for ci, col in enumerate(frac_cols):
        vals = top[col].to_numpy(dtype=float)
        stage = col.replace("frac_stage_", "")
        ax.bar(x, vals, bottom=bottom, label=stage, width=0.7,
               color=_BAR_COLORS[ci % len(_BAR_COLORS)], edgecolor="none")
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of predicted variance")
    ax.set_xlabel("Top uncertain variants (ranked)")
    ax.set_title("Per-Variant Stage Mix (Top Uncertain)")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    return _fig_to_base64(fig)


def _plot_shrunk_vs_unshrunk(df: pd.DataFrame) -> str | None:
    if "fitness_shrunk" not in df.columns or "fitness_avg" not in df.columns:
        return None
    x = df["fitness_avg"].to_numpy(dtype=float)
    y = df["fitness_shrunk"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return None
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(x[valid], y[valid], s=8, alpha=0.4, color=_PALETTE["purple"], edgecolors="none")
    mn = float(min(np.min(x[valid]), np.min(y[valid])))
    mx = float(max(np.max(x[valid]), np.max(y[valid])))
    ax.plot([mn, mx], [mn, mx], color=_PALETTE["red"], ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("Unshrunk fitness (fitness_avg)")
    ax.set_ylabel("Shrunk fitness (fitness_shrunk)")
    ax.set_title("Shrinkage Map: Unshrunk vs Shrunk")
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Status badge helper
# ---------------------------------------------------------------------------
def _status_badge(status: str) -> str:
    s = status.lower().strip()
    if s in ("ok", "success", "converged"):
        cls = "badge-ok"
    elif s in ("warn", "warning"):
        cls = "badge-warn"
    else:
        cls = "badge-error"
    return f'<span class="badge {cls}">{status}</span>'


# ---------------------------------------------------------------------------
# Main report writer
# ---------------------------------------------------------------------------

def write_error_model_report_html(
    report_path: str,
    model_name: str,
    status: str,
    message: str,
    run_metadata: Dict[str, Any],
    parameters: Dict[str, float],
    diagnostics: Dict[str, Any],
    variant_df: pd.DataFrame,
    artifacts: Dict[str, str],
) -> None:
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load mascot
    fish_b64 = _load_mr_fish()
    mascot_tag = (
        f'<img class="mascot" src="data:image/png;base64,{fish_b64}" alt="Mr Fish"/>'
        if fish_b64 else ""
    )
    footer_mascot = (
        f'<img class="mascot-sm" src="data:image/png;base64,{fish_b64}" alt="Mr Fish"/>'
        if fish_b64 else ""
    )

    # Generate plots
    plot_unc = _plot_uncertainty_vs_count(variant_df)
    plot_var = _plot_observed_vs_predicted(variant_df)

    top_uncertain = variant_df.sort_values("fitness_sigma", ascending=False).head(25)
    cols = [
        c
        for c in [
            "REFERENCE_ID",
            "AA_MUTATIONS",
            "fitness_avg",
            "fitness_shrunk",
            "fitness_shrinkage_delta",
            "fitness_sigma",
            "dominant_error_stage",
            "fitness_ci_lower",
            "fitness_ci_upper",
            "stage_attribution_quality_flag",
        ]
        if c in top_uncertain.columns
    ]
    top_uncertain_html = (
        '<div class="table-wrapper">'
        + top_uncertain[cols].to_html(index=False, classes="table")
        + "</div>"
        if cols
        else '<p class="text-secondary text-sm">No variant table columns available.</p>'
    )

    var_stage_cols = _stage_columns(variant_df, "var_stage_")
    frac_stage_cols = _stage_columns(variant_df, "frac_stage_")

    stage_summary = diagnostics.get("stage_global_summary", {}) if isinstance(diagnostics, dict) else {}
    stage_summary_df = pd.DataFrame(stage_summary).T if stage_summary else pd.DataFrame()
    stage_summary_html = (
        '<div class="table-wrapper">' + stage_summary_df.to_html(classes="table") + "</div>"
        if not stage_summary_df.empty
        else '<p class="text-secondary text-sm">No stage summary available.</p>'
    )

    stage_plot = _plot_global_stage_fractions(stage_summary)
    stage_plot_html = _wrap_plot(stage_plot) if stage_plot else '<p class="text-secondary text-sm">No global stage plot available.</p>'
    model_flow = _plot_model_flow(model_name)
    model_flow_html = _wrap_plot(model_flow)
    stage_formula = _plot_stage_variance_composition()
    stage_formula_html = _wrap_plot(stage_formula)
    top_stage_plot = _plot_stage_top_variants(variant_df)
    top_stage_plot_html = _wrap_plot(top_stage_plot) if top_stage_plot else '<p class="text-secondary text-sm">No top-variant stage mix plot available.</p>'

    dom_plot = _plot_dominant_stage_distribution(variant_df)
    dom_plot_html = _wrap_plot(dom_plot) if dom_plot else '<p class="text-secondary text-sm">No dominant-stage distribution available.</p>'
    conf_plot = _plot_stage_confidence(variant_df)
    conf_plot_html = _wrap_plot(conf_plot) if conf_plot else '<p class="text-secondary text-sm">No stage-confidence distribution available.</p>'
    shrink_plot = _plot_shrinkage_delta(variant_df)
    shrink_plot_html = _wrap_plot(shrink_plot) if shrink_plot else '<p class="text-secondary text-sm">No shrinkage output columns available.</p>'
    shrink_scatter = _plot_shrunk_vs_unshrunk(variant_df)
    shrink_scatter_html = _wrap_plot(shrink_scatter) if shrink_scatter else '<p class="text-secondary text-sm">No shrunk-vs-unshrunk scatter available.</p>'
    shrink_flow = _plot_shrinkage_workflow()
    shrink_flow_html = _wrap_plot(shrink_flow)

    stage_variant_html = '<p class="text-secondary text-sm">No per-variant stage attribution columns detected.</p>'
    if var_stage_cols or frac_stage_cols:
        stage_cols = [c for c in ["REFERENCE_ID", "AA_MUTATIONS", "dominant_error_stage"] + var_stage_cols + frac_stage_cols if c in top_uncertain.columns]
        if stage_cols:
            stage_variant_html = '<div class="table-wrapper">' + top_uncertain[stage_cols].to_html(index=False, classes="table") + "</div>"

    params_json = json.dumps(parameters, indent=2)
    diag_json = json.dumps(diagnostics, indent=2)

    # Build key-value metadata cards
    meta_items = ""
    for k, v in run_metadata.items():
        meta_items += f'<div class="kv-item"><div class="kv-label">{k}</div><div class="kv-value">{v}</div></div>'

    artifact_lines = ""
    for k, v in artifacts.items():
        artifact_lines += f'<li><code>{k}</code> <span class="text-secondary">{v}</span></li>'

    var_plot_html = (
        f'<h3>Observed vs Predicted Variance</h3>{_wrap_plot(plot_var)}'
        if plot_var
        else '<p class="text-secondary text-sm">No observed/predicted variance diagnostics for this model.</p>'
    )

    ident = diagnostics.get("identifiability", {}) if isinstance(diagnostics, dict) else {}
    ident_html = '<p class="text-secondary text-sm">No identifiability diagnostics.</p>'
    if ident:
        ident_html = f"""
        <ul class="ident-list">
          <li><b>Lower-bound parameters:</b> {ident.get('n_lower_bound_params', 'n/a')}</li>
          <li><b>Indices at lower bound:</b> {ident.get('lower_bound_param_indices', [])}</li>
          <li><b>Hessian condition (approx):</b> {ident.get('hessian_condition', 'n/a')}</li>
        </ul>
        """

    stage_glossary_rows = ""
    for stage, desc in STAGE_DESCRIPTIONS.items():
        stage_glossary_rows += f"<tr><td><code>{stage}</code></td><td>{desc}</td></tr>"
    stage_glossary_html = f"<table><thead><tr><th>Stage</th><th>Interpretation</th></tr></thead><tbody>{stage_glossary_rows}</tbody></table>"

    status_badge = _status_badge(status)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Error Model Report &mdash; DMS Librarian</title>
  <style>{CSS}</style>
</head>
<body>
<div class="page-wrapper">

  <!-- Header -->
  <div class="report-header">
    {mascot_tag}
    <div class="header-text">
      <h1>Error Model Report</h1>
      <div class="subtitle">Powered by <span class="brand">DMS Librarian</span> &amp; Mr&nbsp;Fish</div>
    </div>
  </div>
  <div class="timestamp">Generated {datetime.utcnow().isoformat()}Z</div>

  <!-- Navigation -->
  <div class="nav-pills">
    <a href="#summary">Summary</a>
    <a href="#how-it-works">How It Works</a>
    <a href="#parameters">Parameters</a>
    <a href="#global-stage">Global Stage Attribution</a>
    <a href="#diagnostics">Diagnostics</a>
    <a href="#per-variant">Per-Variant Attribution</a>
    <a href="#shrinkage">Shrinkage</a>
    <a href="#uncertain">Most Uncertain</a>
    <a href="#artifacts">Artifacts</a>
  </div>

  <!-- 1. Run Summary -->
  <div class="card" id="summary">
    <h2>Run Summary</h2>
    <p class="section-desc">Overview of this error-model run.</p>
    <div class="kv-grid">
      <div class="kv-item">
        <div class="kv-label">Model</div>
        <div class="kv-value">{model_name}</div>
      </div>
      <div class="kv-item">
        <div class="kv-label">Status</div>
        <div class="kv-value">{status_badge}</div>
      </div>
      <div class="kv-item">
        <div class="kv-label">Message</div>
        <div class="kv-value">{message}</div>
      </div>
      {meta_items}
    </div>
  </div>

  <!-- 2. How This Error Model Works -->
  <div class="card" id="how-it-works">
    <h2>How This Error Model Works</h2>
    <p class="section-desc">The model predicts per-variant uncertainty by decomposing replicate variance into biologically meaningful experimental stages, then combines them into total <code>fitness_sigma</code>.</p>
    {model_flow_html}
    <h3>Stage Variance Composition</h3>
    <p class="text-secondary text-sm">For each variant, total variance is the sum of six stage-specific terms. Fitted parameters (<code>phi_*</code>, <code>kappa_*</code>, sorting/dictionary rates) control each term as a function of count depth.</p>
    {stage_formula_html}
    <details>
      <summary>Show variance equations</summary>
      <div class="details-body">
        <pre>V_total = V_library + V_prep + V_sort + V_dict + V_seq + V_residual
fitness_sigma = sqrt(V_total)
frac_stage_s = V_stage_s / V_total</pre>
      </div>
    </details>
    <h3>Stage Glossary</h3>
    <div class="table-wrapper">{stage_glossary_html}</div>
  </div>

  <!-- 3. Model Parameters -->
  <div class="card" id="parameters">
    <h2>Model Parameters</h2>
    <p class="section-desc">Fitted parameter values for the error model.</p>
    <details open>
      <summary>Parameter JSON</summary>
      <div class="details-body">
        <pre>{params_json}</pre>
      </div>
    </details>
  </div>

  <!-- 4. Global Stage Attribution -->
  <div class="card" id="global-stage">
    <h2>Global Stage Attribution</h2>
    <p class="section-desc">How total variance is distributed across experimental stages, averaged over all variants.</p>
    {stage_plot_html}
    {stage_summary_html}
  </div>

  <!-- 5. Diagnostics -->
  <div class="card" id="diagnostics">
    <h2>Diagnostics</h2>
    <p class="section-desc">Convergence, identifiability, and model-fit diagnostics.</p>
    <details>
      <summary>Full diagnostics JSON</summary>
      <div class="details-body">
        <pre>{diag_json}</pre>
      </div>
    </details>
    {ident_html}
    {var_plot_html}
    <h3>Uncertainty vs Input Count</h3>
    {_wrap_plot(plot_unc)}
  </div>

  <!-- 6. Per-Variant Stage Attribution -->
  <div class="card" id="per-variant">
    <h2>Per-Variant Stage Attribution</h2>
    <p class="section-desc">Stage decomposition for the 25 most uncertain variants.</p>
    {stage_variant_html}
    <h3>Per-Variant Stage Mix (Top Uncertain)</h3>
    {top_stage_plot_html}
    <h3>Stage Confidence Distribution</h3>
    {conf_plot_html}
    <h3>Dominant Stage Distribution</h3>
    {dom_plot_html}
  </div>

  <!-- 7. Shrinkage Effects -->
  <div class="card" id="shrinkage">
    <h2>Shrinkage Effects</h2>
    <p class="section-desc">Empirical Bayes shrinkage pulls noisy estimates toward the global mean.</p>
    <h3>Shrinkage Workflow</h3>
    {shrink_flow_html}
    <details>
      <summary>Show shrinkage equations</summary>
      <div class="details-body">
        <pre>Observation: fitness_avg ~ N(theta, sigma^2)
Prior: theta ~ N(mu0, tau^2)
weight_obs = tau^2 / (tau^2 + sigma^2)
fitness_shrunk = weight_obs * fitness_avg + (1 - weight_obs) * mu0</pre>
      </div>
    </details>
    {shrink_scatter_html}
    {shrink_plot_html}
  </div>

  <!-- 8. Most Uncertain Variants -->
  <div class="card" id="uncertain">
    <h2>Most Uncertain Variants</h2>
    <p class="section-desc">Top 25 variants ranked by fitness_sigma.</p>
    {top_uncertain_html}
  </div>

  <!-- 9. Artifacts -->
  <div class="card" id="artifacts">
    <h2>Artifacts</h2>
    <p class="section-desc">Output files generated alongside this report.</p>
    <ul class="artifact-list">{artifact_lines}</ul>
  </div>

  <!-- Footer -->
  <div class="report-footer">
    {footer_mascot}
    <p>DMS Librarian &mdash; Error Model Report</p>
  </div>

</div>
</body>
</html>"""

    out_path.write_text(html)


# ---------------------------------------------------------------------------
# Fitness Analysis Report
# ---------------------------------------------------------------------------

def _embed_png(path: Path) -> str | None:
    """Read a PNG file and return a base64-encoded string, or None."""
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode("ascii")
    return None


def _plot_or_placeholder(b64: str | None, alt: str) -> str:
    if b64:
        return f'<div class="plot-container"><img src="data:image/png;base64,{b64}" alt="{alt}"/></div>'
    return f'<p class="text-secondary text-sm">Plot not available: {alt}</p>'


def _dual_plot_section(
    dir_path: Path,
    base_name: str,
    alt: str,
    has_shrunk: bool,
) -> str:
    """Render an unshrunk plot, and optionally the shrunk version in a toggle."""
    unshrunk = _embed_png(dir_path / f"{base_name}.png")
    html = _plot_or_placeholder(unshrunk, alt)
    if has_shrunk:
        shrunk = _embed_png(dir_path / f"{base_name}_shrunk.png")
        if shrunk:
            html += """
<details>
  <summary>Show shrunk version</summary>
  <div class="details-body">
""" + _plot_or_placeholder(shrunk, f"{alt} (shrunk)") + """
  </div>
</details>"""
    return html


def write_fitness_report_html(
    report_path: str,
    output_dir: str,
    df: pd.DataFrame,
    input_pools: list[str],
    output_pools: list[str],
    fitness_cols: list[str],
    min_input: int,
    input_csv: str,
    has_shrunk: bool = False,
    error_model_name: str | None = None,
    group_by_reference: bool = False,
) -> None:
    """Write an Apple-styled, Mr-Fish-branded HTML report for the full
    fitness analysis (all plots + summary statistics)."""

    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dir_path = Path(output_dir)

    # Mascot
    fish_b64 = _load_mr_fish()
    mascot_tag = (
        f'<img class="mascot" src="data:image/png;base64,{fish_b64}" alt="Mr Fish"/>'
        if fish_b64 else ""
    )
    footer_mascot = (
        f'<img class="mascot-sm" src="data:image/png;base64,{fish_b64}" alt="Mr Fish"/>'
        if fish_b64 else ""
    )

    # ---- Summary statistics ------------------------------------------------
    n_total = len(df)
    fitness_avg = df["fitness_avg"] if "fitness_avg" in df.columns else pd.Series(dtype=float)
    has_hamming = "hamming" in df.columns

    # Variant counts by Hamming distance
    hamming_rows = ""
    if has_hamming:
        for h in sorted(df["hamming"].unique()):
            cnt = int((df["hamming"] == h).sum())
            hamming_rows += f"<tr><td>{h}</td><td>{cnt:,}</td></tr>"

    # Mutation type breakdown (for Hamming-1 singles)
    n_stops = int(df["has_stop"].sum()) if "has_stop" in df.columns else 0
    n_pro = int(df["has_proline"].sum()) if "has_proline" in df.columns else 0
    n_other = int(df["other"].sum()) if "other" in df.columns else 0

    # WT fitness
    wt_rows = df[df["AA_MUTATIONS"] == "WT"] if "AA_MUTATIONS" in df.columns else pd.DataFrame()
    wt_fitness = f"{float(wt_rows['fitness_avg'].iloc[0]):.4f}" if (len(wt_rows) > 0 and "fitness_avg" in wt_rows.columns) else "n/a"

    # Reference templates
    refs = sorted(df["REFERENCE_ID"].unique().tolist()) if "REFERENCE_ID" in df.columns else []

    # Replicate correlations
    corr_rows = ""
    for i in range(len(fitness_cols)):
        for j in range(i + 1, len(fitness_cols)):
            valid = df[[fitness_cols[i], fitness_cols[j]]].dropna()
            if len(valid) > 1:
                r = float(np.corrcoef(valid[fitness_cols[i]], valid[fitness_cols[j]])[0, 1])
                corr_rows += f"<tr><td>Replicate {i+1} vs {j+1}</td><td>{r:.4f}</td></tr>"

    # ---- Embed plots -------------------------------------------------------
    mutability_html = _dual_plot_section(dir_path, "mutability_plot", "Mutability Plot", has_shrunk)
    epistasis_html = _dual_plot_section(dir_path, "epistasis_plot", "Epistasis Plot", has_shrunk)
    fitness_dist_html = _dual_plot_section(dir_path, "fitness_distributions", "Fitness Distributions", has_shrunk)
    hamming_dist_html = _dual_plot_section(dir_path, "hamming_distributions", "Hamming Distance Distributions", has_shrunk)
    reproducibility_html = _dual_plot_section(dir_path, "reproducibility_plot", "Reproducibility Plot", has_shrunk)
    subst_matrix_html = _dual_plot_section(dir_path, "substitution_matrix", "Substitution Matrix", has_shrunk)

    # ---- Top/bottom variants table -----------------------------------------
    top_bottom_html = ""
    if not fitness_avg.empty:
        top10 = df.nlargest(10, "fitness_avg")
        bot10 = df.nsmallest(10, "fitness_avg")
        show_cols = [c for c in ["AA_MUTATIONS", "fitness_avg", "fitness_shrunk", "fitness_sigma", "hamming"] if c in df.columns]
        if show_cols:
            top_tbl = top10[show_cols].to_html(index=False, classes="table", float_format="%.4f")
            bot_tbl = bot10[show_cols].to_html(index=False, classes="table", float_format="%.4f")
            top_bottom_html = f"""
<h3>Top 10 Most Beneficial</h3>
<div class="table-wrapper">{top_tbl}</div>
<h3>Top 10 Most Deleterious</h3>
<div class="table-wrapper">{bot_tbl}</div>"""

    # ---- Metadata cards ----------------------------------------------------
    meta_items = f"""
<div class="kv-item"><div class="kv-label">Input CSV</div><div class="kv-value">{Path(input_csv).name}</div></div>
<div class="kv-item"><div class="kv-label">Variants</div><div class="kv-value">{n_total:,}</div></div>
<div class="kv-item"><div class="kv-label">Replicates</div><div class="kv-value">{len(fitness_cols)}</div></div>
<div class="kv-item"><div class="kv-label">Min Input</div><div class="kv-value">{min_input}</div></div>
<div class="kv-item"><div class="kv-label">Input Pools</div><div class="kv-value">{', '.join(input_pools)}</div></div>
<div class="kv-item"><div class="kv-label">Output Pools</div><div class="kv-value">{', '.join(output_pools)}</div></div>
<div class="kv-item"><div class="kv-label">WT Fitness</div><div class="kv-value">{wt_fitness}</div></div>"""
    if error_model_name:
        meta_items += f'<div class="kv-item"><div class="kv-label">Error Model</div><div class="kv-value">{error_model_name}</div></div>'
    if has_shrunk:
        meta_items += '<div class="kv-item"><div class="kv-label">Shrinkage</div><div class="kv-value">Enabled</div></div>'
    if refs:
        meta_items += f'<div class="kv-item"><div class="kv-label">References</div><div class="kv-value">{", ".join(refs)}</div></div>'

    # ---- Fitness summary stats ---------------------------------------------
    fitness_stats_html = ""
    if not fitness_avg.empty:
        fitness_stats_html = f"""
<div class="kv-grid">
  <div class="kv-item"><div class="kv-label">Mean</div><div class="kv-value">{fitness_avg.mean():.4f}</div></div>
  <div class="kv-item"><div class="kv-label">Median</div><div class="kv-value">{fitness_avg.median():.4f}</div></div>
  <div class="kv-item"><div class="kv-label">Std Dev</div><div class="kv-value">{fitness_avg.std():.4f}</div></div>
  <div class="kv-item"><div class="kv-label">Min</div><div class="kv-value">{fitness_avg.min():.4f}</div></div>
  <div class="kv-item"><div class="kv-label">Max</div><div class="kv-value">{fitness_avg.max():.4f}</div></div>
  <div class="kv-item"><div class="kv-label">% Beneficial (&gt;0)</div><div class="kv-value">{(fitness_avg > 0).mean():.1%}</div></div>
</div>"""

    # ---- Assemble HTML -----------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Fitness Analysis Report &mdash; DMS Librarian</title>
  <style>{CSS}</style>
</head>
<body>
<div class="page-wrapper">

  <!-- Header -->
  <div class="report-header">
    {mascot_tag}
    <div class="header-text">
      <h1>Fitness Analysis Report</h1>
      <div class="subtitle">Powered by <span class="brand">DMS Librarian</span> &amp; Mr&nbsp;Fish</div>
    </div>
  </div>
  <div class="timestamp">Generated {datetime.utcnow().isoformat()}Z</div>

  <!-- Navigation -->
  <div class="nav-pills">
    <a href="#summary">Summary</a>
    <a href="#statistics">Statistics</a>
    <a href="#mutability">Mutability</a>
    <a href="#fitness-dist">Fitness Distributions</a>
    <a href="#hamming">Hamming Distance</a>
    <a href="#epistasis">Epistasis</a>
    <a href="#reproducibility">Reproducibility</a>
    <a href="#substitution">Substitution Matrix</a>
    <a href="#top-variants">Top Variants</a>
  </div>

  <!-- 1. Run Summary -->
  <div class="card" id="summary">
    <h2>Run Summary</h2>
    <p class="section-desc">Overview of the fitness analysis run and dataset.</p>
    <div class="kv-grid">{meta_items}</div>
  </div>

  <!-- 2. Statistics -->
  <div class="card" id="statistics">
    <h2>Fitness Statistics</h2>
    <p class="section-desc">Summary statistics for average fitness across all variants.</p>
    {fitness_stats_html}

    <h3>Mutation Type Breakdown</h3>
    <div class="table-wrapper">
      <table>
        <thead><tr><th>Category</th><th>Count</th></tr></thead>
        <tbody>
          <tr><td>Stop codons</td><td>{n_stops:,}</td></tr>
          <tr><td>Proline mutations</td><td>{n_pro:,}</td></tr>
          <tr><td>Other mutations</td><td>{n_other:,}</td></tr>
        </tbody>
      </table>
    </div>

    {"<h3>Variants by Hamming Distance</h3><div class='table-wrapper'><table><thead><tr><th>Hamming</th><th>Count</th></tr></thead><tbody>" + hamming_rows + "</tbody></table></div>" if hamming_rows else ""}

    {"<h3>Replicate Correlations</h3><div class='table-wrapper'><table><thead><tr><th>Pair</th><th>Pearson r</th></tr></thead><tbody>" + corr_rows + "</tbody></table></div>" if corr_rows else ""}
  </div>

  <!-- 3. Mutability -->
  <div class="card" id="mutability">
    <h2>Mutability</h2>
    <p class="section-desc">Average fitness effect at each amino acid position for single mutants (Hamming 1), relative to wild type. Green bars are beneficial; red bars are deleterious.</p>
    {mutability_html}
  </div>

  <!-- 4. Fitness Distributions -->
  <div class="card" id="fitness-dist">
    <h2>Fitness Distributions</h2>
    <p class="section-desc">Kernel density estimates of fitness for single mutants, separated by mutation type: stop codons, proline substitutions, and all other amino acid changes.</p>
    {fitness_dist_html}
  </div>

  <!-- 5. Hamming Distance -->
  <div class="card" id="hamming">
    <h2>Hamming Distance Distributions</h2>
    <p class="section-desc">Fitness distributions stratified by the number of amino acid substitutions (Hamming distance from wild type). Shows how fitness landscapes shift as mutations accumulate.</p>
    {hamming_dist_html}
  </div>

  <!-- 6. Epistasis -->
  <div class="card" id="epistasis">
    <h2>Epistasis</h2>
    <p class="section-desc">Comparison of observed double-mutant fitness against the additive expectation (sum of the two constituent single-mutant effects). Points above the diagonal indicate positive epistasis; below indicates negative epistasis.</p>
    {epistasis_html}
  </div>

  <!-- 7. Reproducibility -->
  <div class="card" id="reproducibility">
    <h2>Reproducibility</h2>
    <p class="section-desc">Pairwise replicate correlation heatmaps. Each panel shows the 2D density of fitness values between two replicates, with the Pearson correlation coefficient in the title.</p>
    {reproducibility_html}
  </div>

  <!-- 8. Substitution Matrix -->
  <div class="card" id="substitution">
    <h2>Substitution Matrix</h2>
    <p class="section-desc">Average fitness for every wild-type to mutant amino acid substitution, arranged by biochemical similarity. Red indicates beneficial substitutions; blue indicates deleterious. Annotations show the fitness value and sample count.</p>
    {subst_matrix_html}
  </div>

  <!-- 9. Top Variants -->
  <div class="card" id="top-variants">
    <h2>Top Variants</h2>
    <p class="section-desc">The ten most beneficial and ten most deleterious variants by average fitness.</p>
    {top_bottom_html}
  </div>

  <!-- Footer -->
  <div class="report-footer">
    {footer_mascot}
    <p>DMS Librarian &mdash; Fitness Analysis Report</p>
  </div>

</div>
</body>
</html>"""

    out_path.write_text(html)
