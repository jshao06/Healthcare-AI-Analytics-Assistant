"""Orchestration entry point for deterministic DataFrame analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .insights import generate_insights
from .profiler import profile_dataframe
from .statistics import (
    calculate_categorical_statistics,
    calculate_correlations,
    calculate_missing_values,
    calculate_numeric_statistics,
)
from .types import AnalysisResult, ColumnStatistics


class DataAnalyzer:
    """Coordinate profiling, statistics, and rule-based insight generation."""

    def __init__(
        self,
        *,
        high_cardinality_threshold: int | float = 0.5,
        missing_threshold: float = 10.0,
        skew_threshold: float = 1.0,
        correlation_threshold: float = 0.8,
    ) -> None:
        self._high_cardinality_threshold = high_cardinality_threshold
        self._missing_threshold = missing_threshold
        self._skew_threshold = skew_threshold
        self._correlation_threshold = correlation_threshold

    def analyze(self, dataframe: object) -> AnalysisResult:
        """Run the complete analysis pipeline for a pandas DataFrame."""

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")

        profile = profile_dataframe(
            dataframe,
            high_cardinality_threshold=self._high_cardinality_threshold,
        )

        numeric = calculate_numeric_statistics(dataframe)
        categorical = calculate_categorical_statistics(dataframe)
        missing = calculate_missing_values(dataframe)
        correlations = calculate_correlations(dataframe)
        statistics = _build_column_statistics(dataframe, numeric, categorical)

        insights = generate_insights(
            dataframe,
            missing_threshold=self._missing_threshold,
            skew_threshold=self._skew_threshold,
            correlation_threshold=self._correlation_threshold,
            numeric_statistics=numeric,
            missing_values=missing,
            correlations=correlations,
        )

        return AnalysisResult(
            dataframe_summary=profile,
            statistics=statistics,
            insights=insights,
        )


def _build_column_statistics(
    dataframe: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
) -> tuple[ColumnStatistics, ...]:
    statistics: list[ColumnStatistics] = []
    numeric_rows = dict(numeric.iterrows())
    categorical_rows = dict(categorical.iterrows())

    for column, series in dataframe.items():
        if column in numeric_rows:
            row = numeric_rows[column]
            statistics.append(
                ColumnStatistics(
                    column=str(column),
                    data_type=str(series.dtype),
                    minimum=_python_scalar(row["minimum"]),
                    maximum=_python_scalar(row["maximum"]),
                    mean=float(row["mean"]),
                    standard_deviation=float(row["standard_deviation"]),
                    median=float(row["median"]),
                    missing_count=int(row["missing_count"]),
                )
            )
            continue

        row = categorical_rows[column]
        statistics.append(
            ColumnStatistics(
                column=str(column),
                data_type=str(series.dtype),
                missing_count=int(row["missing_count"]),
            )
        )

    return tuple(statistics)


def _python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item() # pyright: ignore[reportUnknownVariableType]
    return value


__all__ = ["DataAnalyzer"]
