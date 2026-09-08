"""Render backend-neutral chart specifications as Matplotlib figures."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from matplotlib.figure import Figure

from ..analysis.types import Chart
from .types import ChartType


class _ChartAxes(Protocol):
    """Matplotlib Axes operations required by this renderer."""

    def bar(self, x: object, height: object) -> object: ...

    def tick_params(self, *, axis: str, labelrotation: float) -> None: ...

    def plot(self, x: object, y: object, *, marker: str) -> object: ...

    def scatter(self, x: object, y: object) -> object: ...

    def set_xticks(self, ticks: object) -> object: ...

    def set_title(self, label: str) -> object: ...

    def set_xlabel(self, xlabel: str) -> object: ...

    def set_ylabel(self, ylabel: str) -> object: ...


class ChartBuilder:
    """Create a Matplotlib figure from a chart specification."""

    def build(self, specification: object) -> Figure:
        if not isinstance(specification, Chart):
            raise TypeError("specification must be an analysis Chart")
        try:
            chart_type = ChartType(specification.chart_type)
        except ValueError as error:
            raise ValueError(f"unsupported chart type: {specification.chart_type}") from error
        if not specification.data:
            raise ValueError("chart specification contains no data")

        labels = list(specification.data)
        raw_x_values = list(specification.data[labels[0]])
        raw_y_values = list(specification.data[labels[1]]) if len(labels) > 1 else []
        figure = Figure(figsize=(8, 5), layout="constrained")
        axes = cast(_ChartAxes, figure.subplots())

        if chart_type is ChartType.BAR:
            _require_pair(labels, raw_x_values, raw_y_values)
            x_values = np.asarray(raw_x_values)
            y_values = np.asarray(raw_y_values)
            axes.bar(x_values, y_values)
            axes.tick_params(axis="x", labelrotation=45)
        elif chart_type is ChartType.LINE:
            _require_pair(labels, raw_x_values, raw_y_values)
            x_values = np.asarray(raw_x_values)
            y_values = np.asarray(raw_y_values)
            axes.plot(x_values, y_values, marker="o")
            if len(x_values) > 10:
                axes.set_xticks(x_values[:: max(1, len(x_values) // 10)])
            axes.tick_params(axis="x", labelrotation=45)
        elif chart_type is ChartType.SCATTER:
            _require_pair(labels, raw_x_values, raw_y_values)
            x_values = np.asarray(raw_x_values)
            y_values = np.asarray(raw_y_values)
            axes.scatter(x_values, y_values)
            axes.tick_params(axis="x", labelrotation=45)

        axes.set_title(specification.title)
        axes.set_xlabel(specification.x_label or labels[0])
        axes.set_ylabel(specification.y_label or labels[1])
        return figure


def _require_pair(labels: list[str], x_values: list[object], y_values: list[object]) -> None:
    if len(labels) < 2:
        raise ValueError("chart specification requires two data series")
    if len(x_values) != len(y_values):
        raise ValueError("chart data series must have equal lengths")


__all__ = ["ChartBuilder"]
