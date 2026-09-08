"""Models for user-facing assistant responses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from ..sql.executor import QueryResult as ExecutorQueryResult


@dataclass(frozen=True, slots=True)
class Warning:
    """A non-fatal issue to show with an assistant response."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class QueryResult:
    """A serializable snapshot of a successful query result."""

    columns: tuple[str, ...]
    rows: tuple[dict[str, object], ...]
    row_count: int
    execution_time_ms: float
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    """An error safe to present to the user."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    """The final response returned by the assistant."""

    message: str
    result: QueryResult | None = None
    error: ErrorResponse | None = None
    warnings: tuple[Warning, ...] = field(default_factory=tuple)


class ResponseBuilder:
    """Build response models from query results or user-facing errors."""

    @staticmethod
    def from_query_result(
        result: ExecutorQueryResult,
        *,
        message: str = "Query completed successfully.",
        warnings: Sequence[Warning] = (),
    ) -> AssistantResponse:
        """Build a successful assistant response."""

        records = cast(
            list[dict[str, object]],
            result.data.to_dict(orient="records"),
        )
        query_result = QueryResult(
            columns=result.columns,
            rows=tuple(records),
            row_count=result.row_count,
            execution_time_ms=result.execution_time_ms,
            truncated=result.truncated,
        )
        return AssistantResponse(
            message=message,
            result=query_result,
            warnings=tuple(warnings),
        )

    @staticmethod
    def from_error(
        code: str,
        message: str,
        *,
        warnings: Sequence[Warning] = (),
    ) -> AssistantResponse:
        """Build an assistant response containing a user-facing error."""

        return AssistantResponse(
            message=message,
            error=ErrorResponse(code=code, message=message),
            warnings=tuple(warnings),
        )
