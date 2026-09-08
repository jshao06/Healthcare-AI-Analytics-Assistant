"""Main orchestration for the healthcare analytics assistant."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..sql.executor import (
    QueryResult as ExecutorQueryResult,
    SQLExecutionError,
    SQLExecutor,
    SQLValidationError,
)
from ..sql.generator import SQLGenerationError, SQLGenerator
from ..sql.types import SQLQueryRequest
from ..sql.validator import SQLValidator, ValidationIssue, ValidationResult
from .context import Context, ContextBuilder
from .response import AssistantResponse, ResponseBuilder, Warning


class QueryRequestBuilder(Protocol):
    """Convert a question and prompt context into a structured SQL request."""

    def build(self, question: str, context: Context) -> SQLQueryRequest:
        """Return the structured request consumed by ``SQLGenerator``."""
        raise NotImplementedError


class HealthcareAssistant:
    """Coordinate question processing through final response creation."""

    def __init__(
        self,
        context_builder: ContextBuilder,
        request_builder: QueryRequestBuilder,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executor: SQLExecutor,
        response_builder: ResponseBuilder | None = None,
    ) -> None:
        self._context_builder = context_builder
        self._request_builder = request_builder
        self._sql_generator = sql_generator
        self._sql_validator = sql_validator
        self._sql_executor = sql_executor
        self._response_builder = response_builder or ResponseBuilder()

    def ask(self, question: str) -> AssistantResponse:
        """Run one natural-language question through the complete workflow."""

        normalized_question = question.strip()
        if not normalized_question:
            return self._response_builder.from_error(
                code="invalid_question",
                message="Please provide a healthcare analytics question.",
            )

        try:
            context = self.build_context(normalized_question)
        except ValueError:
            return self._response_builder.from_error(
                code="context_error",
                message="The assistant could not build database context.",
            )

        try:
            sql, parameters = self.generate_sql(
                normalized_question,
                context,
            )
        except (SQLGenerationError, ValueError, TypeError):
            return self._response_builder.from_error(
                code="sql_generation_error",
                message="The assistant could not generate a valid SQL query.",
            )

        validation = self.validate_sql(sql)
        if not validation.is_valid:
            return self._response_builder.from_error(
                code="sql_validation_error",
                message=self._validation_message(validation.issues),
                warnings=self._validation_warnings(validation.issues),
            )

        warnings = list(self._validation_warnings(validation.issues))

        try:
            result = self.execute_sql(sql, parameters)
        except SQLExecutionError as error:
            # The error has been converted to a safe response. Drop its
            # database traceback so failed SQLite cursors are not retained.
            error.__traceback__ = None
            error.__cause__ = None
            return self._response_builder.from_error(
                code="sql_execution_error",
                message="The database could not complete the query.",
                warnings=warnings,
            )
        except SQLValidationError:
            return self._response_builder.from_error(
                code="sql_validation_error",
                message="The generated SQL did not pass validation.",
                warnings=warnings,
            )

        if result.truncated:
            warnings.append(Warning(
                code="result_truncated",
                message="The result was limited to the maximum number of rows.",
            ))

        return self.build_response(result, warnings=warnings)

    def build_context(self, question: str) -> Context:
        """Build database and clinical context for a question."""

        return self._context_builder.build(question)

    def generate_sql(
        self,
        question: str,
        context: Context,
    ) -> tuple[str, list[object]]:
        """Create structured query intent and render parameterized SQL."""

        request = self._request_builder.build(question, context)
        return self._sql_generator.generate(request)

    def validate_sql(self, sql: str) -> ValidationResult:
        """Validate generated SQL before database execution."""

        return self._sql_validator.validate(sql)

    def execute_sql(
        self,
        sql: str,
        parameters: Sequence[object],
    ) -> ExecutorQueryResult:
        """Execute validated SQL with separately bound parameters."""

        return self._sql_executor.execute(sql, parameters)

    def build_response(
        self,
        result: ExecutorQueryResult,
        *,
        warnings: Sequence[Warning] = (),
    ) -> AssistantResponse:
        """Convert an executor result into the final assistant response."""

        return self._response_builder.from_query_result(
            result,
            warnings=warnings,
        )

    @staticmethod
    def _validation_message(issues: list[ValidationIssue]) -> str:
        if not issues:
            return "The generated SQL did not pass validation."
        return " ".join(issue.message for issue in issues)

    @staticmethod
    def _validation_warnings(
        issues: list[ValidationIssue],
    ) -> tuple[Warning, ...]:
        return tuple(
            Warning(code=issue.code, message=issue.message)
            for issue in issues
        )


# Short compatibility name retained for existing imports.
Assistant = HealthcareAssistant

