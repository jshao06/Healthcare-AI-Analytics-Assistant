"""Deterministic statistical helpers for tabular analysis.

The functions in this module deliberately operate only on pandas objects and
return pandas/NumPy values.  They do not infer meaning from column names or use
an AI service.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _validate_dataframe(dataframe: object) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")


def _numeric_columns(dataframe: pd.DataFrame) -> list[object]:
    """Return numeric columns, excluding booleans."""

    return [
        column
        for column in dataframe.columns
        if is_numeric_dtype(dataframe[column].dtype)
        and not is_bool_dtype(dataframe[column].dtype)
    ]


def _as_float_array(series: pd.Series) -> npt.NDArray[np.float64]:
    """Convert a numeric Series to a NumPy array with missing values as NaN."""

    values: list[float] = []
    for value, missing in zip(series.tolist(), series.isna().tolist(), strict=True):
        if missing:
            values.append(float("nan"))
            continue

        if isinstance(value, complex):
            if abs(value.imag) < 1e-12:
                value = value.real
            else:
                values.append(float("nan"))
                continue

        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def calculate_numeric_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics for every numeric column.

    The returned frame is indexed by the original column names and contains
    ``count``, ``missing_count``, ``mean``, ``standard_deviation``, ``minimum``,
    ``25%``, ``median``, ``75%`` and ``maximum``.  Standard deviation uses the
    sample convention (``ddof=1``), matching :meth:`pandas.Series.std`.
    """

    _validate_dataframe(dataframe)
    columns = _numeric_columns(dataframe)
    result_columns = [
        "count",
        "missing_count",
        "mean",
        "standard_deviation",
        "minimum",
        "25%",
        "median",
        "75%",
        "maximum",
    ]
    if not columns:
        return pd.DataFrame(columns=result_columns, index=pd.Index([], name="column"))

    numeric = dataframe.loc[:, columns]
    rows: list[dict[str, int | float]] = []
    for _, series in numeric.items():
        values = _as_float_array(series)
        valid = values[~np.isnan(values)]
        count = int(valid.size)
        rows.append(
            {
                "count": count,
                "missing_count": int(values.size) - count,
                "mean": float(np.mean(valid)) if count else float("nan"),
                "standard_deviation": (
                    float(np.std(valid, ddof=1)) if count > 1 else float("nan")
                ),
                "minimum": float(np.min(valid)) if count else float("nan"),
                "25%": float(np.quantile(valid, 0.25)) if count else float("nan"),
                "median": float(np.median(valid)) if count else float("nan"),
                "75%": float(np.quantile(valid, 0.75)) if count else float("nan"),
                "maximum": float(np.max(valid)) if count else float("nan"),
            }
        )
    result = pd.DataFrame(rows, index=numeric.columns, columns=result_columns)
    result.index.name = "column"
    return result[result_columns]


def calculate_categorical_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate counts, cardinality and mode for non-numeric columns.

    Missing values are excluded when finding the most frequent value.  Tied
    modes are resolved deterministically by pandas' value order (first seen).
    """

    _validate_dataframe(dataframe)
    numeric_columns = set(_numeric_columns(dataframe))
    columns = [column for column in dataframe.columns if column not in numeric_columns]
    result_columns = ["count", "missing_count", "unique_count", "most_frequent", "frequency"]
    rows: list[dict[str, object]] = []

    for column in columns:
        series = dataframe[column]
        frequencies = series.value_counts(dropna=True, sort=True)
        rows.append(
            {
                "count": int(series.count()),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "most_frequent": frequencies.index[0] if not frequencies.empty else None,
                "frequency": int(frequencies.iloc[0]) if not frequencies.empty else 0,
            }
        )

    result = pd.DataFrame(rows, index=pd.Index(columns, name="column"), columns=result_columns)
    return result


def calculate_correlations(
    dataframe: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Return the pairwise correlation matrix for numeric columns."""

    _validate_dataframe(dataframe)
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")
    if min_periods < 1:
        raise ValueError("min_periods must be a positive integer")
    numeric = dataframe.loc[:, _numeric_columns(dataframe)]
    return numeric.corr(method=method, min_periods=min_periods)


def calculate_distributions(
    dataframe: pd.DataFrame,
    bins: int | Sequence[float] | str = 10,
) -> dict[object, pd.DataFrame]:
    """Calculate a histogram for each numeric column.

    Each mapping value has ``bin_left``, ``bin_right`` and ``count`` columns.
    Missing and infinite values are omitted.  Empty columns produce an empty
    frame instead of raising an exception.
    """

    _validate_dataframe(dataframe)
    if isinstance(bins, bool) or (isinstance(bins, int) and bins < 1):
        raise ValueError("bins must be a positive integer, a sequence of edges, or a NumPy bin rule")

    distributions: dict[object, pd.DataFrame] = {}
    numeric = dataframe.loc[:, _numeric_columns(dataframe)]
    for column, series in numeric.items():
        values = _as_float_array(series)
        values = values[np.isfinite(values)]
        if values.size == 0:
            distributions[column] = pd.DataFrame(columns=["bin_left", "bin_right", "count"])
            continue
        counts, edges = np.histogram(values, bins=bins)
        distributions[column] = pd.DataFrame(
            {"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts.astype("int64")}
        )
    return distributions


def calculate_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages for every column."""

    _validate_dataframe(dataframe)
    counts = np.fromiter(
        (
            np.count_nonzero(np.asarray(series.isna().tolist(), dtype=np.bool_))
            for _, series in dataframe.items()
        ),
        dtype=np.int64,
        count=len(dataframe.columns),
    )
    denominator = len(dataframe)
    percentages = counts.astype(np.float64)
    if denominator:
        percentages *= 100.0 / denominator
    result = pd.DataFrame(
        {"missing_count": counts, "missing_percentage": percentages},
        index=dataframe.columns,
    )
    result.index.name = "column"
    return result


__all__ = [
    "calculate_numeric_statistics",
    "calculate_categorical_statistics",
    "calculate_correlations",
    "calculate_distributions",
    "calculate_missing_values",
]
