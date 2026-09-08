"""
executor.py

Validates and executes read-only SQL queries and returns structured results.

Author: Jianhua Shao
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from ..db.database import Database
from .validator import SQLValidator, ValidationIssue


@dataclass(frozen=True, eq=False)
class QueryResult:
    """Result returned after successful SQL execution."""

    sql: str
    data: pd.DataFrame
    execution_time_ms: float
    truncated: bool

    @property
    def columns(self) -> tuple[str, ...]:
        """Return current result column names."""
        return tuple(str(column) for column in self.data.columns)

    @property
    def row_count(self) -> int:
        """Return the current number of result rows."""
        return len(self.data)

    @property
    def is_empty(self) -> bool:
        """Return whether the query returned no rows."""
        return self.row_count == 0


class SQLExecutionError(RuntimeError):
    """Raised when the database cannot execute a SQL query."""


class SQLValidationError(ValueError):
    """Raised when SQL does not pass the configured validator."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        codes = ", ".join(issue.code for issue in self.issues)
        super().__init__(f"SQL validation failed: {codes}")


class SQLExecutor:
    """Execute validated SQL queries against the configured database.

    Parameters
    ----------
    database:
        Database interface used to execute SQL.
    validator:
        Validator that enforces the read-only SQL policy before execution.
    max_rows:
        Maximum number of rows returned to the caller.

    """

    def __init__(
        self,
        database: Database,
        validator: SQLValidator,
        max_rows: int = 500,
    ) -> None:
        if max_rows < 1:
            raise ValueError('max_rows must be at least 1')

        self._database = database
        self._validator = validator
        self._max_rows = max_rows

    def execute(
        self,
        sql: str,
        parameters: Sequence[object] = (),
    ) -> QueryResult:
        """Execute SQL and return a structured query result.

        Parameters
        ----------
        sql:
            A SQL statement to validate and execute.
        parameters:
            Values bound to SQL placeholders by the database adapter.

        Returns
        -------
        QueryResult
            Query data and execution metadata.

        Raises
        ------
        SQLValidationError
            If SQL does not pass validation.
        SQLExecutionError
            If the database cannot execute the query.
        """
        validation = self._validator.validate(sql)
        if not validation.is_valid:
            raise SQLValidationError(validation.issues)

        normalized_sql = validation.sql

        start_time = perf_counter()

        try:
            dataframe = self._database.query(
                normalized_sql,
                parameters,
                max_rows=self._max_rows + 1,
            )
        except Exception as exc:
            raise SQLExecutionError(
                'The database could not execute the query.'
            ) from exc

        execution_time_ms = (perf_counter() - start_time) * 1000

        truncated = len(dataframe) > self._max_rows

        if truncated:
            dataframe = dataframe.iloc[:self._max_rows].copy()

        return QueryResult(
            sql=normalized_sql,
            data=dataframe,
            execution_time_ms=execution_time_ms,
            truncated=truncated,
        )
