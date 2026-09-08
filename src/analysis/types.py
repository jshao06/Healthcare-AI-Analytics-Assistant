"""Shared data structures for healthcare result analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


def _empty_string_mapping() -> Mapping[str, str]:
    """Return an immutable, correctly typed empty mapping."""

    return MappingProxyType({})


class InsightSeverity(str, Enum):
    """Importance level assigned to an analysis insight."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ColumnStatistics:
    """Descriptive statistics calculated for one DataFrame column."""

    column: str
    data_type: str
    minimum: object | None = None
    maximum: object | None = None
    mean: float | None = None
    standard_deviation: float | None = None
    median: float | None = None
    missing_count: int = 0


@dataclass(frozen=True, slots=True)
class DataProfile:
    """High-level structural summary of a DataFrame."""

    row_count: int
    column_count: int
    duplicate_count: int
    memory_usage_bytes: int
    missing_percentage: float = 0.0
    data_types: Mapping[str, str] = field(default_factory=_empty_string_mapping)
    high_cardinality_columns: tuple[str, ...] = field(default_factory=tuple)
    constant_columns: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Insight:
    """A user-facing observation derived from analyzed data."""

    title: str
    description: str
    severity: InsightSeverity = InsightSeverity.INFO
    recommendation: str = ""


@dataclass(frozen=True, slots=True)
class Chart:
    """Serializable chart specification produced by analysis."""

    title: str
    chart_type: str
    data: Mapping[str, Sequence[object]]
    x_label: str | None = None
    y_label: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Complete output returned by the analysis package."""

    dataframe_summary: DataProfile
    statistics: tuple[ColumnStatistics, ...] = field(default_factory=tuple)
    insights: tuple[Insight, ...] = field(default_factory=tuple)
    charts: tuple[Chart, ...] = field(default_factory=tuple)
