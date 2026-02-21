#!/usr/bin/env python3
"""Generate synthetic merged_on_nonsyn_counts datasets for error model testing."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def make_aa_mutation(rng: np.random.Generator, idx: int) -> str:
    # Keep WT at row 0
    if idx == 0:
        return "WT"

    n_mut = int(rng.choice([1, 2], p=[0.8, 0.2]))
    muts = []
    used_pos = set()
    for _ in range(n_mut):
        pos = int(rng.integers(1, 301))
        while pos in used_pos:
            pos = int(rng.integers(1, 301))
        used_pos.add(pos)
        wt = rng.choice(AA_LIST)
        mut = rng.choice(AA_LIST + ["*"])
        muts.append(f"{wt}{pos}{mut}")
    return "+".join(muts)


def simulate_counts(
    n_variants: int,
    n_reps: int,
    seed: int,
    base_input_reads_per_rep: int,
    wt_frac: float,
    phi_lib: float,
    phi_sort: float,
    p_fp: float,
    p_fn: float,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)

    # Library frequencies (WT + skewed tail)
    alpha = np.full(n_variants, 0.25)
    alpha[0] = 5.0  # WT enriched
    raw_freq = rng.dirichlet(alpha)

    # Force WT closer to requested fraction and renormalize
    raw_freq[1:] *= (1.0 - wt_frac) / raw_freq[1:].sum()
    raw_freq[0] = wt_frac

    # True fitness around WT, with heavier deleterious tail
    true_fitness = rng.normal(loc=-0.25, scale=0.65, size=n_variants)
    true_fitness[0] = 0.0
    # Beneficial subpopulation
    beneficial_idx = rng.choice(np.arange(1, n_variants), size=max(1, n_variants // 20), replace=False)
    true_fitness[beneficial_idx] += rng.uniform(0.5, 1.5, size=len(beneficial_idx))

    rows = []
    for i in range(n_variants):
        row = {
            "REFERENCE_ID": "refA",
            "AA_MUTATIONS": make_aa_mutation(rng, i),
            "CONSENSUS_IDS": f"cons_{i:05d}",
            "NUC_MUTATIONS": "",
        }

        for r in range(1, n_reps + 1):
            # Input replicate scale variability (library prep + sequencing overdispersion proxy)
            rep_scale_in = np.exp(rng.normal(0.0, phi_lib))
            mu_in = max(0.1, base_input_reads_per_rep * raw_freq[i] * rep_scale_in)
            in_count = rng.poisson(mu_in)

            # Selection model with gate error and sort overdispersion proxy
            sel_prob_true = 1.0 / (1.0 + np.exp(-(true_fitness[i] + rng.normal(0, phi_sort))))
            sel_prob_obs = sel_prob_true * (1.0 - p_fn) + (1.0 - sel_prob_true) * p_fp
            sel_prob_obs = min(max(sel_prob_obs, 1e-6), 1 - 1e-6)

            # Output count roughly proportional to selected fraction and input
            # with extra multiplicative noise
            rep_scale_out = np.exp(rng.normal(0.0, phi_sort))
            mu_out = max(0.05, in_count * sel_prob_obs * 1.8 * rep_scale_out)
            out_count = rng.poisson(mu_out)

            row[f"in{r}"] = int(in_count)
            row[f"out{r}"] = int(out_count)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure WT has healthy counts in all replicates
    for r in range(1, n_reps + 1):
        if df.loc[0, f"in{r}"] < 50:
            df.loc[0, f"in{r}"] = 50
        if df.loc[0, f"out{r}"] < 20:
            df.loc[0, f"out{r}"] = 20

    truth = {
        "seed": seed,
        "n_variants": n_variants,
        "n_replicates": n_reps,
        "base_input_reads_per_rep": base_input_reads_per_rep,
        "wt_fraction": wt_frac,
        "phi_lib": phi_lib,
        "phi_sort": phi_sort,
        "p_fp": p_fp,
        "p_fn": p_fn,
        "notes": "Synthetic droplet-style counts for testing error models",
    }

    return df, truth


def write_dataset(base_dir: Path, stem: str, df: pd.DataFrame, truth: dict) -> None:
    csv_path = base_dir / f"{stem}.csv"
    truth_path = base_dir / f"{stem}.truth.json"
    yaml_path = base_dir / f"{stem}.droplet_full.yaml"

    df.to_csv(csv_path, index=False)
    truth_path.write_text(json.dumps(truth, indent=2))

    # Example model config for this dataset
    yaml_path.write_text(
        "\n".join(
            [
                "model: droplet_full",
                "fit_error_model: true",
                "n_bootstraps: 200",
                "ci_level: 0.95",
                "single_rep_mode: fallback",
                "random_seed: 42",
                "report_html:",
                "  enabled: true",
                "params:",
                "  prior_strength: 1.0",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "data" / "error_model_synthetic"
    root.mkdir(parents=True, exist_ok=True)

    # Main 3-replicate dataset
    df3, truth3 = simulate_counts(
        n_variants=1200,
        n_reps=3,
        seed=20260221,
        base_input_reads_per_rep=1_600_000,
        wt_frac=0.012,
        phi_lib=0.30,
        phi_sort=0.22,
        p_fp=0.015,
        p_fn=0.03,
    )
    write_dataset(root, "merged_on_nonsyn_counts.synthetic_reps3", df3, truth3)

    # Single-replicate fallback dataset
    df1, truth1 = simulate_counts(
        n_variants=900,
        n_reps=1,
        seed=20260222,
        base_input_reads_per_rep=900_000,
        wt_frac=0.015,
        phi_lib=0.38,
        phi_sort=0.28,
        p_fp=0.02,
        p_fn=0.05,
    )
    write_dataset(root, "merged_on_nonsyn_counts.synthetic_single_rep", df1, truth1)

    print(f"Wrote synthetic datasets to: {root}")
