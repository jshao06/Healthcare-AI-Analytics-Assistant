"""Data profiling utilities for healthcare analytics results."""

from __future__ import annotations

from types import MappingProxyType

import pandas as pd

from .types import DataProfile


def profile_dataframe(
    dataframe: object,
    *,
    high_cardinality_threshold: int | float = 0.5,
) -> DataProfile:
    """Produce a lightweight structural profile of a DataFrame.

    ``high_cardinality_threshold`` may be either an absolute unique-value
    count (an integer) or a fraction of all rows (a float from zero to
    one). A column is high-cardinality when its unique count is strictly above
    the resolved threshold. Missing values count as a distinct value when
    identifying constant columns.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    _validate_high_cardinality_threshold(high_cardinality_threshold)

    row_count, column_count = dataframe.shape
    total_cells = row_count * column_count
    missing_count = int(dataframe.isna().sum().sum())
    missing_percentage = missing_count * 100.0 / total_cells if total_cells else 0.0

    data_types = MappingProxyType(
        {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}
    )
    high_cardinality_columns: list[str] = []
    constant_columns: list[str] = []

    for column, series in dataframe.items():
        unique_count = int(series.nunique(dropna=True))
        unique_with_missing = int(series.nunique(dropna=False))
        if unique_with_missing <= 1:
            constant_columns.append(str(column))

        threshold = _resolve_cardinality_threshold(
            high_cardinality_threshold,
            int(row_count),
        )
        if unique_count > threshold:
            high_cardinality_columns.append(str(column))

    return DataProfile(
        row_count=int(row_count),
        column_count=int(column_count),
        duplicate_count=int(dataframe.duplicated().sum()),
        memory_usage_bytes=int(dataframe.memory_usage(index=True, deep=True).sum()),
        missing_percentage=missing_percentage,
        data_types=data_types,
        high_cardinality_columns=tuple(high_cardinality_columns),
        constant_columns=tuple(constant_columns),
    )


def _validate_high_cardinality_threshold(threshold: int | float) -> None:
    if isinstance(threshold, bool):
        raise ValueError("high_cardinality_threshold must be a positive number")
    if isinstance(threshold, int):
        if threshold < 1:
            raise ValueError("an integer high_cardinality_threshold must be at least 1")
        return
    if not 0.0 < threshold <= 1.0:
        raise ValueError("a float high_cardinality_threshold must be in the range (0, 1]")


def _resolve_cardinality_threshold(threshold: int | float, row_count: int) -> float:
    if isinstance(threshold, int):
        return float(threshold)
    return threshold * row_count


__all__ = ["profile_dataframe"]
