"""CSV loading and column validation for the ISIC 2024 dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def load_data(path: str | Path) -> pd.DataFrame:
    """Load ISIC 2024 metadata CSV and log basic statistics.

    Args:
        path: Path to train-metadata.csv or test-metadata.csv.

    Returns:
        Raw DataFrame with no modifications.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, low_memory=False)

    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    if "target" in df.columns:
        counts = df["target"].value_counts(dropna=False)
        logger.info(f"Target distribution (discovered):\n{counts.to_string()}")
        n_pos = int(counts.get(1, 0))
        n_total = len(df)
        logger.info(
            f"Positive rate: {n_pos}/{n_total} = {n_pos / n_total * 100:.4f}%  "
            f"(imbalance ratio ≈ {(n_total - n_pos) / max(n_pos, 1):.0f}:1)"
        )

    return df


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required columns are missing.

    Args:
        df: DataFrame to validate.
        required: Column names that must be present.

    Raises:
        ValueError: If one or more required columns are absent.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )
