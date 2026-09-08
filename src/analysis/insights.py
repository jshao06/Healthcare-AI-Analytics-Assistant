"""Generate simple deterministic insights from analysis results."""

from __future__ import annotations

import math
from itertools import combinations

import pandas as pd

from .statistics import (
    calculate_correlations,
    calculate_missing_values,
    calculate_numeric_statistics,
)
from .types import Insight, InsightSeverity


def generate_insights(
    dataframe: pd.DataFrame,
    *,
    missing_threshold: float = 10.0,
    correlation_threshold: float = 0.8,
    skew_threshold: float = 1.0,
    numeric_statistics: pd.DataFrame | None = None,
    missing_values: pd.DataFrame | None = None,
    correlations: pd.DataFrame | None = None,
) -> tuple[Insight, ...]:
    """Generate a small set of useful findings from a DataFrame."""

    if dataframe.empty:
        return ()

    _validate_thresholds(
        missing_threshold,
        correlation_threshold,
        skew_threshold,
    )

    if numeric_statistics is None:
        numeric_statistics = calculate_numeric_statistics(dataframe)
    if missing_values is None:
        missing_values = calculate_missing_values(dataframe)
    if correlations is None:
        correlations = calculate_correlations(dataframe)

    insights: list[Insight] = []

    insights.extend(_average_insights(numeric_statistics))
    insights.extend(_missing_insights(missing_values, missing_threshold))
    insights.extend(_skew_insights(dataframe, skew_threshold))
    insights.extend(_correlation_insights(correlations, correlation_threshold))

    return tuple(insights)


def _average_insights(
    statistics: pd.DataFrame,
) -> list[Insight]:
    """Generate simple average-value observations."""

    insights: list[Insight] = []

    for column, row in statistics.iterrows():
        column_name = str(column)

        if _is_identifier(column_name):
            continue

        mean = _coerce_float(row["mean"])

        if math.isnan(mean):
            continue

        label = _humanize(column_name)
        unit = _column_unit(column_name)

        insights.append(
            Insight(
                title=f"Average {label.lower()}",
                description=(
                    f"Average {label.lower()} is "
                    f"{_format_number(mean)}{unit}."
                ),
            )
        )

    return insights


def _missing_insights(
    statistics: pd.DataFrame,
    threshold: float,
) -> list[Insight]:
    """Report columns with meaningful missing data."""

    insights: list[Insight] = []

    for column, row in statistics.iterrows():
        percentage = _coerce_float(row["missing_percentage"])

        if percentage < threshold:
            continue

        label = _humanize(str(column))

        severity = (
            InsightSeverity.CRITICAL
            if percentage >= 50.0
            else InsightSeverity.WARNING
        )

        insights.append(
            Insight(
                title=f"Missing {label.lower()} values",
                description=(
                    f"{label} has "
                    f"{_format_number(percentage)}% "
                    "missing values."
                ),
                severity=severity,
            )
        )

    return insights


def _correlation_insights(
    correlations: pd.DataFrame,
    threshold: float,
) -> list[Insight]:
    """Report strong relationships between numeric columns."""

    insights: list[Insight] = []
    columns = list(correlations.columns)

    for left, right in combinations(columns, 2):
        if _is_identifier(str(left)) or _is_identifier(str(right)):
            continue

        raw_value = correlations.loc[left, right]
        value = _coerce_float(raw_value)

        if math.isnan(value) or abs(value) < threshold:
            continue

        left_label = _humanize(str(left))
        right_label = _humanize(str(right))

        direction = "negative" if value < 0 else "positive"

        insights.append(
            Insight(
                title=(
                    f"High correlation: "
                    f"{left_label} and {right_label}"
                ),
                description=(
                    f"{left_label} and {right_label} "
                    f"have a strong {direction} "
                    f"correlation ({_format_number(value)}) "
                    f"between {left_label.lower()} and {right_label.lower()}."
                ),
            )
        )

    return insights


def _is_identifier(column: str) -> bool:
    """Return whether a column appears to be an identifier."""

    tokens = (
        column.lower()
        .replace("-", "_")
        .split("_")
    )

    return bool(
        set(tokens)
        & {"id", "identifier", "code"}
    )


def _humanize(column: str) -> str:
    """Convert a database column name into readable text."""

    text = column.replace("_", " ").strip()

    if not text:
        return "Column"

    return text[0].upper() + text[1:]


def _column_unit(column: str) -> str:
    """Return a human-friendly unit label for common numeric columns."""

    key = column.lower()
    if "age" in key or "year" in key:
        return " years"
    return ""


def _coerce_float(value: object) -> float:
    """Normalize numpy/pandas scalar objects to a plain float."""

    if value is None:
        return math.nan

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, complex):
        return float(value.real) if abs(value.imag) < 1e-12 else math.nan

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return math.nan

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _coerce_float(item())
        except (AttributeError, TypeError, ValueError):
            return math.nan

    return math.nan


def _format_number(value: float) -> str:
    """Format a number without unnecessary trailing zeros."""

    return f"{value:.2f}".rstrip("0").rstrip(".")


def _skew_insights(
    dataframe: pd.DataFrame,
    threshold: float,
) -> list[Insight]:
    """Surface strong right-skew in numeric columns."""

    insights: list[Insight] = []

    for column in dataframe.columns:
        if _is_identifier(str(column)):
            continue

        series = dataframe[column]
        if not pd.api.types.is_numeric_dtype(series.dtype):
            continue

        value = _coerce_float(series.skew())
        if math.isnan(value) or value <= threshold:
            continue

        label = _humanize(str(column))
        insights.append(
            Insight(
                title=f"{label} is skewed",
                description=(
                    f"{label} has strong right skew "
                    f"({_format_number(value)})."
                ),
            )
        )

    return insights


def _validate_thresholds(
    missing_threshold: float,
    correlation_threshold: float,
    skew_threshold: float,
) -> None:
    """Validate insight configuration."""

    if not 0.0 <= missing_threshold <= 100.0:
        raise ValueError(
            "missing_threshold must be between 0 and 100"
        )

    if not 0.0 <= correlation_threshold <= 1.0:
        raise ValueError(
            "correlation_threshold must be between 0 and 1"
        )

    if not 0.0 <= skew_threshold:
        raise ValueError(
            "skew_threshold must be non-negative"
        )


__all__ = ["generate_insights"]