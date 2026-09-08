"""Public types used by the visualization package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from matplotlib.figure import Figure

from ..analysis.types import Chart


class ChartType(str, Enum):
    """Chart families supported by the visualization package."""

    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"


@dataclass(frozen=True, slots=True)
class ChartOptions:
    """Optional instructions for selecting and building a chart."""

    chart_type: ChartType = ChartType.BAR
    x: str | None = None
    y: str | None = None
    title: str | None = None
    bins: int = 10
    max_categories: int = 20

    def __post_init__(self) -> None:
        if self.bins < 1:
            raise ValueError("bins must be a positive integer")

        if self.max_categories < 1:
            raise ValueError(
                "max_categories must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    """A selected chart specification and rendered figure."""

    specification: Chart
    figure: Figure


__all__ = [
    "ChartOptions",
    "ChartType",
    "VisualizationResult",
]
