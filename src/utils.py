"""Utility helpers for Fraud Shield AI.

This module centralizes repeatable engineering concerns such as logging,
reproducibility, and artifact persistence.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class RunContext:
    """Metadata for one training/evaluation execution."""

    run_id: str
    started_at_utc: str
    random_state: int


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and pandas scalar types gracefully."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure consistent console logging for scripts and notebooks."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger so modules can log uniformly."""

    return logging.getLogger(name)


def ensure_directories(paths: Iterable[Path | str]) -> None:
    """Create directories if they do not already exist."""

    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = DEFAULT_RANDOM_STATE) -> None:
    """Set all major random seeds for deterministic runs.

    Note: XGBoost and threaded tree methods can still have minimal non-determinism
    in some environments, but this setup minimizes variance substantially.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_run_context(random_state: int = DEFAULT_RANDOM_STATE) -> RunContext:
    """Build a run context object used in audit-ready metadata."""

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("run_%Y%m%d_%H%M%S")
    return RunContext(
        run_id=run_id,
        started_at_utc=started_at.isoformat(),
        random_state=random_state,
    )


def save_json(data: dict[str, Any], output_path: Path | str, indent: int = 2) -> None:
    """Persist dictionaries as UTF-8 JSON with robust type handling."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=indent, cls=NumpyEncoder)


def save_dataframe(df: pd.DataFrame, output_path: Path | str, index: bool = False) -> None:
    """Save a dataframe to CSV and ensure parent directory exists."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)


def pct(value: float, digits: int = 2) -> str:
    """Return a human-readable percentage string."""

    return f"{value * 100:.{digits}f}%"
