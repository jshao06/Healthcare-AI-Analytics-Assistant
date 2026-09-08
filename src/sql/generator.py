"""Generate parameterized SQLite SELECT queries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .types import (
    ColumnReference,
    OrderBySpec,
    SQLQueryRequest,
    SelectColumn,
)


AGGREGATE_FUNCTIONS = {"AVG", "COUNT", "MAX", "MIN", "SUM"}
SORT_DIRECTIONS = {"ASC", "DESC"}


class IdentifierQuoter(Protocol):
    """Database behavior needed to quote SQL identifiers."""

    def quote_identifier(self, identifier: str) -> str: ...


class SchemaProvider(Protocol):
    """Schema behavior required by SQLGenerator."""

    @property
    def database(self) -> IdentifierQuoter: ...

    def has_table(self, table_name: str) -> bool: ...

    def get_columns(self, table_name: str) -> list[str]: ...

    def get_primary_key(self, table_name: str) -> str | None: ...


class SQLGenerationError(ValueError):
    """Raised when a query request cannot be safely generated."""


class SQLGenerator:
    """Generate validated, parameterized SELECT statements."""

    def __init__(self, schema_manager: SchemaProvider) -> None:
        self._schema = schema_manager

    def generate(
        self,
        request: SQLQueryRequest,
    ) -> tuple[str, list[object]]:
        """Generate SQL and separately bound parameter values."""

        base_table = request.get("base_table")
        if not base_table:
            raise SQLGenerationError("base_table is required")

        self._validate_table(base_table)

        available_tables = {base_table}

        select_sql = self._build_select(
            request.get("select", []),
            available_tables,
            request.get("distinct", False),
        )

        group_sql = self._build_group_by(
            request.get("group_by", []),
            available_tables,
        )

        order_sql = self._build_order_by(
            request.get("order_by", []),
            available_tables,
        )

        limit_sql = self._build_limit(request.get("limit"))

        clauses = [
            select_sql,
            f"FROM {self._quote(base_table)}",
            group_sql,
            order_sql,
            limit_sql,
        ]

        sql = "\n".join(clause for clause in clauses if clause)

        return sql, []

    def _build_select(
        self,
        columns: Sequence[SelectColumn],
        available_tables: set[str],
        distinct: bool,
    ) -> str:
        """Build the SELECT clause."""

        prefix = "SELECT DISTINCT" if distinct else "SELECT"

        if not columns:
            return f"{prefix} *"

        expressions = [
            self._select_expression(column, available_tables)
            for column in columns
        ]

        return prefix + "\n    " + ",\n    ".join(expressions)

    def _select_expression(
        self,
        item: SelectColumn,
        available_tables: set[str],
    ) -> str:
        """Render one SELECT expression."""

        table = item.get("table")
        column = item.get("column")

        if not column:
            raise SQLGenerationError(
                "SELECT item requires a column"
            )

        if column == "*":
            expression = "*"
        else:
            if not table:
                raise SQLGenerationError(
                    f"Column {column!r} requires a table"
                )

            self._validate_column(
                table,
                column,
                available_tables,
            )

            expression = self._qualified(table, column)

        aggregate = item.get("aggregate")

        if aggregate is not None:
            if aggregate not in AGGREGATE_FUNCTIONS:
                raise SQLGenerationError(
                    f"Unsupported aggregate: {aggregate}"
                )

            if column == "*":
                if aggregate != "COUNT":
                    raise SQLGenerationError(
                        f"{aggregate}(*) is not supported"
                    )

                if not table:
                    raise SQLGenerationError(
                        "Column '*' requires a table for COUNT"
                    )

                primary_key = self._schema.get_primary_key(table)
                if primary_key is None:
                    raise SQLGenerationError(
                        f"Table {table!r} has no primary key for COUNT"
                    )

                expression = self._qualified(table, primary_key)

            if item.get("distinct"):
                expression = f"DISTINCT {expression}"

            expression = f"{aggregate}({expression})"

        alias = item.get("alias")
        if alias:
            expression += f" AS {self._quote(alias)}"

        return expression

    def _build_group_by(
        self,
        columns: Sequence[ColumnReference],
        available_tables: set[str],
    ) -> str:
        """Build GROUP BY."""

        if not columns:
            return ""

        expressions: list[str] = []

        for item in columns:
            table = item["table"]
            column = item["column"]

            self._validate_column(
                table,
                column,
                available_tables,
            )

            expressions.append(
                self._qualified(table, column)
            )

        return "GROUP BY " + ", ".join(expressions)

    def _build_order_by(
        self,
        items: Sequence[OrderBySpec],
        available_tables: set[str],
    ) -> str:
        """Build ORDER BY."""

        if not items:
            return ""

        expressions: list[str] = []

        for item in items:
            table = item["table"]
            column = item["column"]
            direction = item["direction"]

            if direction not in SORT_DIRECTIONS:
                raise SQLGenerationError(
                    f"Unsupported sort direction: {direction}"
                )

            self._validate_column(
                table,
                column,
                available_tables,
            )

            expressions.append(
                f"{self._qualified(table, column)} "
                f"{direction}"
            )

        return "ORDER BY " + ", ".join(expressions)

    @staticmethod
    def _build_limit(limit: int | None) -> str:
        """Build LIMIT."""

        if limit is None:
            return ""

        if limit <= 0:
            raise SQLGenerationError(
                "limit must be positive"
            )

        return f"LIMIT {limit}"

    def _validate_table(self, table: str) -> None:
        if not self._schema.has_table(table):
            raise SQLGenerationError(
                f"Unknown table: {table!r}"
            )

    def _validate_column(
        self,
        table: str,
        column: str,
        available_tables: set[str] | None = None,
    ) -> None:
        self._validate_table(table)

        if (
            available_tables is not None
            and table not in available_tables
        ):
            raise SQLGenerationError(
                f"Table {table!r} is not available in this query"
            )

        if column not in self._schema.get_columns(table):
            raise SQLGenerationError(
                f"Unknown column {column!r} "
                f"in table {table!r}"
            )

    def _quote(self, identifier: str) -> str:
        return self._schema.database.quote_identifier(
            identifier
        )

    def _qualified(
        self,
        table: str,
        column: str,
    ) -> str:
        return (
            f"{self._quote(table)}."
            f"{self._quote(column)}"
        )


__all__ = [
    "SQLGenerationError",
    "SQLGenerator",
]