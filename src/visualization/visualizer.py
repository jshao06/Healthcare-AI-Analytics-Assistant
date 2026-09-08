"""Coordinate chart selection and Matplotlib figure construction."""

from __future__ import annotations

import pandas as pd

from ..analysis.types import AnalysisResult
from .chart_builder import ChartBuilder
from .chart_selector import ChartSelector
from .types import ChartOptions, ChartType, VisualizationResult


class DataVisualizer:
    """Run the selection and rendering stages of visualization."""

    def __init__(
        self,
        selector: ChartSelector | None = None,
        builder: ChartBuilder | None = None,
    ) -> None:
        self._selector = selector or ChartSelector()
        self._builder = builder or ChartBuilder()

    def visualize(
        self,
        dataframe: pd.DataFrame,
        analysis_result: AnalysisResult | None = None,
        *,
        chart_type: ChartType | str = ChartType.BAR,
        x: str | None = None,
        y: str | None = None,
        title: str | None = None,
        bins: int = 10,
        max_categories: int = 20,
    ) -> tuple[VisualizationResult, ...]:
        if dataframe.empty:
            raise ValueError("cannot select a chart for an empty DataFrame")

        try:
            requested_type = ChartType(chart_type)
        except ValueError as error:
            supported = ", ".join(item.value for item in ChartType)
            raise ValueError(f"unsupported chart type; choose one of: {supported}") from error
        specifications = self._selector.select(
            dataframe,
            analysis_result,
            options=ChartOptions(
                chart_type=requested_type,
                x=x,
                y=y,
                title=title,
                bins=bins,
                max_categories=max_categories,
            ),
        )
        return tuple(
            VisualizationResult(specification=specification, figure=self._builder.build(specification))
            for specification in specifications
        )


Visualizer = DataVisualizer


def visualize(
    dataframe: pd.DataFrame,
    analysis_result: AnalysisResult | None = None,
    **kwargs: object,
) -> tuple[VisualizationResult, ...]:
    """Select and render visualizations with the functional API."""

    return DataVisualizer().visualize(dataframe, analysis_result, **kwargs)  # type: ignore[arg-type]


__all__ = ["DataVisualizer", "Visualizer", "visualize"]
