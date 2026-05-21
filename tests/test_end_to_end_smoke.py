from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_main_end_to_end_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data/raw/creditcard.csv"

    if not raw_csv.exists():
        pytest.skip("creditcard.csv not available in data/raw")

    df = pd.read_csv(raw_csv)

    fraud = df[df["Class"] == 1]
    nonfraud = df[df["Class"] == 0].sample(n=7000, random_state=42)
    sample_df = pd.concat([nonfraud, fraud], ignore_index=True).sample(frac=1, random_state=42)

    sample_csv = tmp_path / "creditcard_smoke.csv"
    sample_df.to_csv(sample_csv, index=False)

    out_dir = tmp_path / "outputs"

    cmd = [
        sys.executable,
        str(project_root / "main.py"),
        "--data-path",
        str(sample_csv),
        "--outputs-dir",
        str(out_dir),
        "--random-state",
        "42",
        "--cv-folds",
        "2",
        "--n-iter",
        "2",
        "--max-experiments",
        "2",
        "--threshold-objective",
        "f2",
    ]

    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"main.py failed with code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    comparison_path = out_dir / "metrics/model_comparison_table.csv"
    summary_path = out_dir / "reports/executive_metrics_summary.json"
    roc_plot = out_dir / "plots/roc_curve_top_experiments.png"

    assert comparison_path.exists()
    assert summary_path.exists()
    assert roc_plot.exists()

    comp = pd.read_csv(comparison_path)
    assert not comp.empty
    assert "test_pr_auc" in comp.columns
