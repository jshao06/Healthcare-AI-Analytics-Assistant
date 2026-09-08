from __future__ import annotations

from typing import Literal, TypedDict


AggregateFunction = Literal[
    "AVG",
    "COUNT",
    "MAX",
    "MIN",
    "SUM",
]

ComparisonOperator = Literal[
    "=",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "IN",
    "LIKE",
    "IS NULL",
    "IS NOT NULL",
]

LogicalOperator = Literal[
    "AND",
    "OR",
]

SortDirection = Literal[
    "ASC",
    "DESC",
]


class SelectColumn(TypedDict, total=False):
    """A column or aggregate expression in a SELECT clause."""

    table: str
    column: str
    aggregate: AggregateFunction
    alias: str
    distinct: bool


class ColumnReference(TypedDict):
    """A required table and column reference."""

    table: str
    column: str


class FilterCondition(TypedDict, total=False):
    """A single WHERE-clause condition."""

    table: str
    column: str
    operator: ComparisonOperator
    value: object
    logical_operator: LogicalOperator


class JoinSpec(TypedDict):
    """A table join definition."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str


class OrderBySpec(TypedDict):
    """An ORDER BY expression."""

    table: str
    column: str
    direction: SortDirection


class SQLQueryRequest(TypedDict, total=False):
    """Structured input consumed by SQLGenerator."""

    base_table: str
    distinct: bool
    select: list[SelectColumn]
    group_by: list[ColumnReference]
    order_by: list[OrderBySpec]
    limit: int
