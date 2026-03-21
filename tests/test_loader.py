"""Tests for src/isic2024/data/loader.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from isic2024.data.loader import load_data, validate_columns

# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def test_load_data_reads_csv(tmp_path: Path, synthetic_df: pd.DataFrame) -> None:
    """load_data should return a DataFrame with the same shape as the saved CSV."""
    csv_path = tmp_path / "train.csv"
    synthetic_df.to_csv(csv_path, index=False)

    df = load_data(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == synthetic_df.shape


def test_load_data_missing_file(tmp_path: Path) -> None:
    """load_data should raise FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_data(tmp_path / "nonexistent.csv")


def test_load_data_logs_target_distribution(
    tmp_path: Path, synthetic_df: pd.DataFrame, capsys: pytest.CaptureFixture
) -> None:
    """load_data should not crash and should return correct target distribution."""
    csv_path = tmp_path / "train.csv"
    synthetic_df.to_csv(csv_path, index=False)

    df = load_data(csv_path)

    assert int(df["target"].sum()) == 3          # 3 malignant in synthetic fixture
    assert len(df) == 100


def test_load_data_accepts_path_string(
    tmp_path: Path, synthetic_df: pd.DataFrame
) -> None:
    """load_data should accept both str and Path."""
    csv_path = tmp_path / "train.csv"
    synthetic_df.to_csv(csv_path, index=False)

    df = load_data(str(csv_path))
    assert len(df) == 100


# ---------------------------------------------------------------------------
# validate_columns
# ---------------------------------------------------------------------------

def test_validate_columns_passes_when_all_present(synthetic_df: pd.DataFrame) -> None:
    """validate_columns should not raise when all required cols are present."""
    validate_columns(synthetic_df, ["patient_id", "target", "age_approx"])  # no exception


def test_validate_columns_raises_on_missing(synthetic_df: pd.DataFrame) -> None:
    """validate_columns should raise ValueError listing missing columns."""
    with pytest.raises(ValueError, match="nonexistent_col"):
        validate_columns(synthetic_df, ["patient_id", "nonexistent_col"])


def test_validate_columns_raises_on_multiple_missing(synthetic_df: pd.DataFrame) -> None:
    """validate_columns should list all missing columns in the error message."""
    with pytest.raises(ValueError) as exc_info:
        validate_columns(synthetic_df, ["target", "foo", "bar"])
    msg = str(exc_info.value)
    assert "foo" in msg
    assert "bar" in msg
