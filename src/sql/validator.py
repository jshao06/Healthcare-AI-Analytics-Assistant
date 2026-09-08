"""Validate generated SQL before database execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError


class SchemaProvider(Protocol):
    """Minimum schema behavior required by SQLValidator."""

    def get_table_names(self) -> list[str]:
        """Return available table names."""
        ...

    def get_columns(self, table_name: str) -> list[str]:
        """Return columns for one table."""
        ...


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One SQL validation problem."""

    code: str
    message: str


@dataclass(slots=True)
class ValidationResult:
    """Result of validating one SQL query."""

    sql: str
    issues: list[ValidationIssue] = field(
        default_factory=list[ValidationIssue]
    )

    @property
    def is_valid(self) -> bool:
        """Return True when no validation issues were found."""

        return not self.issues

    def add_error(self, code: str, message: str) -> None:
        """Add a validation error."""

        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
            )
        )


class SQLValidator:
    """Validate simple read-only analytics queries."""

    def __init__(
        self,
        schema: SchemaProvider,
        *,
        dialect: str = "sqlite",
        require_limit: bool = True,
        max_rows: int = 1000,
        allow_select_star: bool = False,
    ) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")

        self._schema = schema
        self._dialect = dialect
        self._require_limit = require_limit
        self._max_rows = max_rows
        self._allow_select_star = allow_select_star

    def validate(self, sql: str) -> ValidationResult:
        """Validate one generated SQL query."""

        sql = sql.strip()
        result = ValidationResult(sql=sql)

        if not sql:
            result.add_error(
                "empty_sql",
                "SQL cannot be empty.",
            )
            return result

        expression = self._parse(sql, result)
        if expression is None:
            return result

        self._validate_select(expression, result)

        if not result.is_valid:
            return result

        table_aliases = self._validate_tables(
            expression,
            result,
        )

        self._validate_columns(
            expression,
            table_aliases,
            result,
        )

        self._validate_select_star(
            expression,
            result,
        )

        self._validate_limit(
            expression,
            result,
        )

        return result

    def _parse(
        self,
        sql: str,
        result: ValidationResult,
    ) -> exp.Expression | None:  # pyright: ignore[reportPrivateImportUsage]
        """Parse one SQL statement."""

        try:
            return cast(
                exp.Expression,  # pyright: ignore[reportPrivateImportUsage]
                parse_one(
                    sql,
                    read=self._dialect,
                ),
            )
        except ParseError as exc:
            result.add_error(
                "syntax_error",
                f"Invalid SQL syntax: {exc}",
            )
            return None

    @staticmethod
    def _validate_select(
        expression: exp.Expression, # pyright: ignore[reportPrivateImportUsage]
        result: ValidationResult,
    ) -> None:
        """Allow SELECT queries only."""

        if not isinstance(expression, exp.Select):
            result.add_error(
                "select_only",
                "Only SELECT queries are allowed.",
            )

    def _validate_tables(
        self,
        expression: exp.Expression,  # pyright: ignore[reportPrivateImportUsage]
        result: ValidationResult,
    ) -> dict[str, str]:
        """Validate referenced tables and return alias mappings."""

        available_tables = {
            name.lower(): name
            for name in self._schema.get_table_names()
        }

        aliases: dict[str, str] = {}

        for table in expression.find_all(exp.Table):
            table_name = table.name

            if not table_name:
                continue

            actual_name = available_tables.get(
                table_name.lower()
            )

            if actual_name is None:
                result.add_error(
                    "unknown_table",
                    f"Unknown table: {table_name}",
                )
                continue

            aliases[table_name.lower()] = actual_name

            alias = table.alias
            if alias:
                aliases[alias.lower()] = actual_name

        return aliases

    def _validate_columns(
        self,
        expression: exp.Expression,  # pyright: ignore[reportPrivateImportUsage]
        table_aliases: dict[str, str],
        result: ValidationResult,
    ) -> None:
        """Validate referenced columns."""

        referenced_tables = set(table_aliases.values())

        column_cache = {
            table: {
                column.lower()
                for column in self._schema.get_columns(table)
            }
            for table in referenced_tables
        }

        for column in expression.find_all(exp.Column):
            column_name = column.name

            if not column_name or column_name == "*":
                continue

            qualifier = column.table

            if qualifier:
                table_name = table_aliases.get(
                    qualifier.lower()
                )

                if table_name is None:
                    result.add_error(
                        "unknown_table_alias",
                        (
                            f"Unknown table or alias "
                            f"{qualifier!r}."
                        ),
                    )
                    continue

                if (
                    column_name.lower()
                    not in column_cache[table_name]
                ):
                    result.add_error(
                        "unknown_column",
                        (
                            f"Unknown column "
                            f"{column_name!r} "
                            f"in table {table_name!r}."
                        ),
                    )

                continue

            self._validate_unqualified_column(
                column_name,
                referenced_tables,
                column_cache,
                result,
            )

    @staticmethod
    def _validate_unqualified_column(
        column_name: str,
        tables: set[str],
        column_cache: dict[str, set[str]],
        result: ValidationResult,
    ) -> None:
        """Validate a column without a table qualifier."""

        matches = [
            table
            for table in tables
            if column_name.lower() in column_cache[table]
        ]

        if not matches:
            result.add_error(
                "unknown_column",
                f"Unknown column: {column_name}",
            )

        elif len(matches) > 1:
            result.add_error(
                "ambiguous_column",
                (
                    f"Column {column_name!r} is ambiguous; "
                    "qualify it with a table name."
                ),
            )

    def _validate_select_star(
        self,
        expression: exp.Expression,  # pyright: ignore[reportPrivateImportUsage]
        result: ValidationResult,
    ) -> None:
        """Reject SELECT * unless explicitly enabled."""

        if self._allow_select_star:
            return

        if expression.find(exp.Star) is not None:
            result.add_error(
                "select_star_not_allowed",
                "SELECT * is not allowed.",
            )

    def _validate_limit(
        self,
        expression: exp.Expression,  # pyright: ignore[reportPrivateImportUsage]
        result: ValidationResult,
    ) -> None:
        """Require a positive LIMIT within the configured maximum."""

        limit = expression.args.get("limit")

        if limit is None:
            if self._require_limit:
                result.add_error(
                    "missing_limit",
                    (
                        "A LIMIT clause is required "
                        f"(maximum {self._max_rows})."
                    ),
                )
            return

        value = limit.expression

        if not isinstance(value, exp.Literal):
            result.add_error(
                "invalid_limit",
                "LIMIT must be an integer.",
            )
            return

        try:
            limit_value = int(value.this)
        except (TypeError, ValueError):
            result.add_error(
                "invalid_limit",
                "LIMIT must be an integer.",
            )
            return

        if limit_value <= 0:
            result.add_error(
                "invalid_limit",
                "LIMIT must be greater than zero.",
            )

        elif limit_value > self._max_rows:
            result.add_error(
                "limit_too_large",
                (
                    f"LIMIT {limit_value} exceeds "
                    f"the maximum of {self._max_rows}."
                ),
            )


__all__ = [
    "SQLValidator",
    "ValidationIssue",
    "ValidationResult",
]