"""Choose a simple visualization for analysis results."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from ..analysis.types import Chart
from .types import ChartOptions, ChartType


@dataclass(frozen=True, slots=True)
class ChartSelectionAssessment:
    """Assessment of a dataframe's likely charting intent."""

    is_categorical_query: bool
    categorical_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    category_column: str | None = None
    category_count: int = 0
    category_limit: int = 20
    category_count_is_small: bool = False
    chart_type: ChartType = ChartType.BAR


class ChartSelector:
    """Select a useful chart from a DataFrame.

    The selector intentionally uses a small set of predictable rules.
    Complex visualization decisions can be added later if needed.
    """

    def __init__(self, *, max_categories: int = 20) -> None:
        if max_categories < 2:
            raise ValueError("max_categories must be at least 2")

        self._max_categories = max_categories

    def select(
        self,
        frame: DataFrame,
        analysis_result: object | None = None,
        *,
        options: ChartOptions | None = None,
    ) -> tuple[Chart, ...]:
        """Return chart specifications when an obvious chart exists."""

        if frame.empty:
            return ()

        if options is not None:
            chart_type = options.chart_type
            x = options.x
            y = options.y
            if x and y and x in frame.columns and y in frame.columns:
                if chart_type is ChartType.BAR:
                    return (self._bar(frame=frame, x=x, y=y),)
                if chart_type is ChartType.LINE:
                    return (self._line(frame=frame, x=x, y=y),)
                if chart_type is ChartType.SCATTER:
                    return (self._scatter(frame=frame, x=x, y=y),)

        columns = list(frame.columns)
        numeric = [column for column in columns if is_numeric_dtype(frame[column])]
        datetime = [column for column in columns if is_datetime64_any_dtype(frame[column])]
        categorical = [
            column
            for column in columns
            if column not in numeric and column not in datetime
        ]

        if len(columns) == 1 and categorical:
            return (self._categorical_counts(frame, categorical[0]),)

        if datetime and numeric:
            return (self._line(frame=frame, x=datetime[0], y=numeric[0]),)

        time_axis = self._time_series_axis(frame, columns)
        if time_axis is not None and numeric:
            metric_columns = [column for column in numeric if column != time_axis]
            y_column = metric_columns[0] if metric_columns else numeric[0]
            return (self._line(frame=frame, x=time_axis, y=y_column),)

        if len(numeric) >= 2 and self._looks_like_time_series(frame, numeric):
            x_col, y_col = self._time_series_pair(frame, numeric)
            return (self._line(frame=frame, x=x_col, y=y_col),)

        category = self._usable_category(frame, categorical)
        if category is not None and numeric:
            return (self._bar(frame=frame, x=category, y=numeric[0]),)

        if len(numeric) == 1:
            return (self._single_numeric_value(frame, numeric[0]),)

        if len(numeric) >= 2:
            return (self._scatter(frame=frame, x=numeric[0], y=numeric[1]),)

        if categorical:
            return (self._categorical_counts(frame, categorical[0]),)

        return ()

    def assess(
        self,
        frame: DataFrame,
        analysis_result: object | None = None,
    ) -> ChartSelectionAssessment:
        """Return a summary of charting intent for a DataFrame."""

        columns = list(frame.columns)
        numeric = tuple(column for column in columns if is_numeric_dtype(frame[column]))
        datetime = tuple(
            column for column in columns if is_datetime64_any_dtype(frame[column])
        )
        categorical = tuple(
            column for column in columns if column not in numeric and column not in datetime
        )

        category = self._usable_category(frame, list(categorical))
        category_count = int(frame[category].nunique(dropna=True)) if category else 0
        category_limit = self._max_categories
        is_small = bool(category and 1 < category_count <= category_limit)

        if datetime and numeric:
            chart_type = ChartType.LINE
        elif category and numeric:
            chart_type = ChartType.BAR
        elif len(numeric) >= 2:
            chart_type = ChartType.SCATTER
        else:
            chart_type = ChartType.BAR

        return ChartSelectionAssessment(
            is_categorical_query=bool(category and numeric) or bool(len(columns) == 1 and categorical),
            categorical_columns=categorical,
            numeric_columns=numeric,
            category_column=category,
            category_count=category_count,
            category_limit=category_limit,
            category_count_is_small=is_small,
            chart_type=chart_type,
        )

    def _usable_category(
        self,
        frame: DataFrame,
        columns: list[str],
    ) -> str | None:
        """Return the first low-cardinality categorical column."""

        for column in columns:
            category_count = frame[column].nunique(dropna=True)
            if 1 < category_count <= self._max_categories:
                return column

        return None

    @staticmethod
    def _looks_like_time_series(frame: DataFrame, numeric_columns: list[str]) -> bool:
        """Return True only for clearly time-like numeric pairs such as year/value data."""

        if len(numeric_columns) < 2:
            return False

        x_col, y_col = numeric_columns[0], numeric_columns[1]
        x_values = frame[x_col].dropna()
        y_values = frame[y_col].dropna()
        if x_values.empty or y_values.empty or len(x_values) != len(y_values):
            return False

        if not pd.api.types.is_numeric_dtype(x_values) or not pd.api.types.is_numeric_dtype(y_values):
            return False

        x_name = x_col.lower()
        time_tokens = ("year", "month", "day", "date", "time", "quarter", "period")
        if any(token in x_name for token in time_tokens):
            return True

        unique_x = x_values.nunique(dropna=True)
        ordered = x_values.sort_values().reset_index(drop=True)
        if unique_x <= 1 or ordered.equals(ordered.unique()):
            return False

        return False

    @staticmethod
    def _time_series_axis(frame: DataFrame, columns: list[str]) -> str | None:
        """Return a year-like column that should be used as the x-axis of a time series."""

        for column in columns:
            values = frame[column].dropna()
            if values.empty:
                continue
            name = column.lower()
            if not any(token in name for token in ("year", "month", "day", "date", "time", "quarter", "period")):
                continue
            if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_string_dtype(values):
                return column
        return None

    @staticmethod
    def _time_series_pair(frame: DataFrame, numeric_columns: list[str]) -> tuple[str, str]:
        """Pick the first numeric column as x and the second as y for a time-series view."""

        return numeric_columns[0], numeric_columns[1]

    @staticmethod
    def _bar(
        *,
        frame: DataFrame,
        x: str,
        y: str,
    ) -> Chart:
        """Build a bar-chart specification."""

        grouped = frame.groupby(x, sort=True, dropna=False)[y].sum()
        friendly_x = _user_friendly_label(x)
        friendly_y = _user_friendly_label(y)
        return Chart(
            title=f"{friendly_y} by {friendly_x}",
            chart_type=ChartType.BAR.value,
            data={
                x: grouped.index.tolist(),
                y: grouped.tolist(),
            },
            x_label=friendly_x,
            y_label=friendly_y,
        )

    @staticmethod
    def _categorical_counts(frame: DataFrame, x: str) -> Chart:
        """Build a count chart for a single categorical column."""

        counts = frame[x].value_counts(dropna=False).sort_index()
        return Chart(
            title=f"{_user_friendly_label(x)} distribution",
            chart_type=ChartType.BAR.value,
            data={
                x: counts.index.tolist(),
                "Count": counts.tolist(),
            },
            x_label=_user_friendly_label(x),
            y_label="Count",
        )

    @staticmethod
    def _single_numeric_value(frame: DataFrame, y: str) -> Chart:
        """Build a simple bar chart for a single numeric summary value."""

        value = frame[y].iloc[0]
        title = _user_friendly_title(y, "Summary")
        return Chart(
            title=title,
            chart_type=ChartType.BAR.value,
            data={
                "Metric": [y],
                y: [value],
            },
            x_label="Metric",
            y_label=_user_friendly_label(y),
        )

    @staticmethod
    def _line(
        *,
        frame: DataFrame,
        x: str,
        y: str,
    ) -> Chart:
        """Build a line-chart specification."""

        ordered = frame.sort_values(x)
        x_values = ordered[x].tolist()
        normalized_x_values = [
            value.isoformat() if hasattr(value, "isoformat") else value
            for value in x_values
        ]

        friendly_x = _user_friendly_label(x)
        friendly_y = _user_friendly_label(y)
        return Chart(
            title=f"{friendly_y} over {friendly_x}",
            chart_type=ChartType.LINE.value,
            data={
                x: normalized_x_values,
                y: ordered[y].tolist(),
            },
            x_label=friendly_x,
            y_label=friendly_y,
        )

    @staticmethod
    def _scatter(
        *,
        frame: DataFrame,
        x: str,
        y: str,
    ) -> Chart:
        """Build a scatter-chart specification."""

        friendly_x = _user_friendly_label(x)
        friendly_y = _user_friendly_label(y)
        return Chart(
            title=f"{friendly_y} vs {friendly_x}",
            chart_type=ChartType.SCATTER.value,
            data={
                x: frame[x].tolist(),
                y: frame[y].tolist(),
            },
            x_label=friendly_x,
            y_label=friendly_y,
        )


def _user_friendly_label(name: str) -> str:
    """Convert raw column names to user-facing labels."""

    label = name.replace("_", " ").strip()
    label = " ".join(part for part in label.split() if part)
    replacements = {
        "anchor age": "Patient age",
        "average anchor age": "Average patient age",
        "original anchor year": "Year",
        "admission year": "Year",
        "record count": "Admissions",
        "length of stay": "Length of stay",
        "patient age": "Patient age",
        "admissions": "Admissions",
        "gender": "Gender",
        "race": "Race",
        "marital status": "Marital status",
        "hospital expire flag": "Hospital expire flag",
    }
    return replacements.get(label.lower(), label.title())


def _user_friendly_title(metric: str, dimension: str) -> str:
    """Create a polished chart title from metric and grouping dimensions."""

    metric_label = _user_friendly_label(metric)
    dimension_label = _user_friendly_label(dimension)
    if dimension_label.lower() == "summary":
        return metric_label
    if dimension_label.lower() in {"metric", "value"}:
        return metric_label
    if metric_label.lower().startswith("average "):
        return metric_label
    return f"{metric_label} by {dimension_label}"


def select_chart_type(
    frame: DataFrame,
    analysis_result: object | None = None,
    *,
    options: ChartOptions | None = None,
) -> tuple[Chart, ...]:
    """Compatibility wrapper for selecting chart specifications."""

    return ChartSelector().select(frame, analysis_result, options=options)


__all__ = ["ChartSelectionAssessment", "ChartSelector", "select_chart_type"]
