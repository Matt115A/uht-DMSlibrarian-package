from pathlib import Path

import pandas as pd

from uht_DMSlibrarian.reporting import write_error_model_report_html


def test_html_report_written(tmp_path: Path):
    report_path = tmp_path / "report.html"
    variant_df = pd.DataFrame(
        {
            "AA_MUTATIONS": ["WT", "A1V"],
            "fitness_avg": [0.0, 0.2],
            "fitness_sigma": [0.05, 0.1],
            "fitness_ci_lower": [-0.1, 0.0],
            "fitness_ci_upper": [0.1, 0.4],
            "mean_input_count": [1000, 100],
            "observed_var_reps": [0.002, 0.004],
            "predicted_var_reps": [0.003, 0.005],
        }
    )

    write_error_model_report_html(
        report_path=str(report_path),
        model_name="dimsum_analog",
        status="ok",
        message="done",
        run_metadata={"n_variants": 2},
        parameters={"a_in_1": 1.2},
        diagnostics={"objective": 0.01},
        variant_df=variant_df,
        artifacts={"fitness_analysis_results.csv": "some/path.csv"},
    )

    assert report_path.exists()
    text = report_path.read_text()
    assert "Error Model Report" in text
    assert "Most Uncertain Variants" in text
