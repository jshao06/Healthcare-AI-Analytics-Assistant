"""Convert natural-language questions into structured SQL requests."""

from __future__ import annotations

import re

from ..sql.types import ColumnReference, SQLQueryRequest, SelectColumn
from .context import ColumnContext, Context, TableContext


class QueryRequestBuilder:

    _AVERAGE_TERMS = {"average", "avg", "mean"}
    _COUNT_TERMS = {"count", "many", "number", "total"}
    _GROUP_TERMS = {"distribution", "breakdown", "grouped", "by", "per"}

    def __init__(self, *, default_limit: int = 100) -> None:
        if default_limit < 1:
            raise ValueError("default_limit must be positive")
        self._default_limit = default_limit

    def build(self, question: str, context: Context) -> SQLQueryRequest:
        """Build a structured SQL request for a question."""

        normalized = " ".join(question.lower().split())
        if not normalized:
            raise ValueError("Question cannot be empty")    

        table = self._find_table(normalized, context)
        if table is None:
            raise ValueError("could not identify a relevant table")

        columns = self._mentioned_columns(normalized, table)
        tokens = set(re.findall(r"[a-z0-9_]+", normalized))

        if tokens & self._AVERAGE_TERMS:
            return self._build_average(table, columns)

        if tokens & self._GROUP_TERMS:
            return self._build_distribution(table, columns)

        if tokens & self._COUNT_TERMS:
            return self._build_count(table)

        if columns:
            return self._build_retrieval(table, columns)

        raise ValueError("unsupported question")

    def _find_table(
        self,
        question: str,
        context: Context,
    ) -> TableContext | None:
        """Return the relevant table with the strongest question match."""

        best_table: TableContext | None = None
        best_score = 0

        for table in context.relevant_tables:
            score = sum(
                self._contains(question, column.name)
                for column in table.columns
            )

            if self._contains(question, table.name):
                score += 1

            if score > best_score:
                best_table = table
                best_score = score

        return best_table

    def _mentioned_columns(
        self,
        question: str,
        table: TableContext,
    ) -> list[ColumnContext]:
        """Return table columns explicitly mentioned in the question."""

        return [
            column
            for column in table.columns
            if self._contains(question, column.name)
        ]

    def _build_average(
        self,
        table: TableContext,
        columns: list[ColumnContext],
    ) -> SQLQueryRequest:
        """Build AVG(column), optionally grouped by a category."""

        measure = next(
             (column for column in columns if self._is_numeric(column)),
             None,
        )
        if measure is None:
            raise ValueError("average requires a numeric column")

        dimension = next( # pyright: ignore[reportUnknownVariableType]
            (column for column in columns if not self._is_numeric(column)), # pyright: ignore[reportUnknownArgumentType] # pyright: ignore[reportUndefinedVariable] # pyright: ignore[reportUndefinedVariable]
            None,
        )

        select: list[SelectColumn] = []

        request: SQLQueryRequest = {
            "base_table": table.name,
            "select": select,
            "limit": self._default_limit,
        }

        if dimension is not None:
            select.append({
                "table": table.name,
                "column": dimension.name,
            })

            group_by: list[ColumnReference] = [{
                "table": table.name,
                "column": dimension.name,
            }]
            request["group_by"] = group_by

        select.append({
            "table": table.name,
            "column": measure.name,
            "aggregate": "AVG",
            "alias": f"average_{measure.name}",
        })

        return request

    def _build_distribution(
        self,
        table: TableContext,
        columns: list[ColumnContext],
    ) -> SQLQueryRequest:
        """Build a grouped count for a categorical column."""

        dimension = next(
            (column for column in columns if not self._is_numeric(column)), # pyright: ignore[reportUnknownMemberType] # pyright: ignore[reportAttributeAccessIssue]
            None,
        )
        if dimension is None:
            raise ValueError("distribution requires a categorical column")

        count_column = self._count_column(table)

        return {
            "base_table": table.name,
            "select": [
                {
                    "table": table.name,
                    "column": dimension.name,
                },
                {
                    "table": table.name,
                    "column": count_column.name, # pyright: ignore[reportUnknownMemberType]
                    "aggregate": "COUNT",
                    "alias": "record_count",
                },
            ],
            "group_by": [{
                "table": table.name,
                "column": dimension.name,
            }],
            "limit": self._default_limit,
        }

    def _build_count(
        self,
        table: TableContext,
    ) -> SQLQueryRequest:
        """Build a simple record-count request."""

        count_column = self._count_column(table)

        return {
            "base_table": table.name,
            "select": [{
                "table": table.name,
                "column": count_column.name,
                "aggregate": "COUNT",
                "alias": "record_count",
            }],
            "limit": self._default_limit,
        }

    def _build_retrieval(
        self,
        table: TableContext,
        columns: list[ColumnContext],
    ) -> SQLQueryRequest:
        """Retrieve explicitly mentioned columns."""

        return {
            "base_table": table.name,
            "select": [
                {
                    "table": table.name,
                    "column": column.name,
                }
                for column in columns
            ],
            "limit": self._default_limit,
        }

    @staticmethod
    def _contains(question: str, identifier: str) -> bool:
        normalized_question = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
        normalized_identifier = " ".join(re.findall(r"[a-z0-9]+", identifier.lower()))

        phrase_match = re.search(
            rf"\b{re.escape(normalized_identifier)}\b",
            normalized_question,
        )
        if phrase_match is not None:
            return True

        question_terms = set(re.findall(r"[a-z0-9]+", normalized_question))
        identifier_terms = set(re.findall(r"[a-z0-9]+", normalized_identifier))
        return bool(identifier_terms & question_terms)

    @staticmethod
    def _is_numeric(column: ColumnContext) -> bool:
        numeric_types = (
            "INT",
            "REAL",
            "FLOAT",
            "DOUBLE",
            "NUMERIC",
            "DECIMAL",
        )
        return any(
            item in column.data_type.upper()
            for item in numeric_types
        )

    @staticmethod
    def _count_column(table: TableContext) -> ColumnContext:
        """Choose a stable column for COUNT()."""

        primary_key = next(
            (column for column in table.columns if column.primary_key),
            None,
        )
        if primary_key is not None:
            return primary_key

        return table.columns[0]

__all__ = ["QueryRequestBuilder"] # pyright: ignore[reportUnsupportedDunderAll]
      