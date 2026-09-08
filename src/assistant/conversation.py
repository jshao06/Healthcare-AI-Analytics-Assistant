"""Conversation history and follow-up context for the assistant."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import TypeVar
from uuid import uuid4

from .response import AssistantResponse

StateValue = TypeVar("StateValue")


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One completed user/assistant exchange."""

    question: str
    contextual_question: str
    response: AssistantResponse
    sql: str | None = None
    parameters: tuple[object, ...] = ()


class Conversation:
    """Maintain bounded history and state across assistant turns."""

    _REFERENCE_PATTERN = re.compile(
        r"\b(those|these|them|they|their|same|previous|above)\b",
        re.IGNORECASE,
    )
    _FOLLOW_UP_PREFIXES = (
        "also ",
        "and ",
        "filter ",
        "now ",
        "only ",
        "sort ",
        "then ",
    )

    def __init__(
        self,
        session_id: str | None = None,
        *,
        max_turns: int = 20,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self.session_id = session_id or uuid4().hex
        self.max_turns = max_turns
        self._turns: list[ConversationTurn] = []
        self._state: dict[str, object] = {}

    @property
    def history(self) -> tuple[ConversationTurn, ...]:
        """Return an immutable snapshot of retained conversation turns."""

        return tuple(self._turns)

    @property
    def last_turn(self) -> ConversationTurn | None:
        """Return the most recent conversation turn, if one exists."""

        return self._turns[-1] if self._turns else None

    @property
    def previous_sql(self) -> str | None:
        """Return the most recent SQL statement stored in the history."""

        for turn in reversed(self._turns):
            if turn.sql is not None:
                return turn.sql
        return None

    @property
    def state(self) -> dict[str, object]:
        """Return a copy of the current session state."""

        return dict(self._state)

    def add_turn(
        self,
        question: str,
        response: AssistantResponse,
        *,
        contextual_question: str | None = None,
        sql: str | None = None,
        parameters: Sequence[object] = (),
    ) -> ConversationTurn:
        """Append a completed turn and enforce the history size limit."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        resolved_question = (
            contextual_question.strip()
            if contextual_question is not None
            else self.contextualize(normalized_question)
        )
        if not resolved_question:
            raise ValueError("contextual_question must not be empty")

        normalized_sql = sql.strip() if sql is not None else None
        turn = ConversationTurn(
            question=normalized_question,
            contextual_question=resolved_question,
            response=response,
            sql=normalized_sql or None,
            parameters=tuple(parameters),
        )
        self._turns.append(turn)

        if len(self._turns) > self.max_turns:
            del self._turns[:-self.max_turns]

        return turn

    def contextualize(self, question: str) -> str:
        """Add prior request/SQL context when a question looks like a follow-up."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        previous = self.last_turn
        if previous is None or not self.is_follow_up(normalized_question):
            return normalized_question

        lines = [
            f"Previous request: {previous.contextual_question}",
        ]
        if self.previous_sql is not None:
            lines.append(f"Previous SQL: {self.previous_sql}")
        lines.append(f"Current follow-up: {normalized_question}")
        return "\n".join(lines)

    def is_follow_up(self, question: str) -> bool:
        """Return whether a question contains conservative follow-up signals."""

        normalized_question = question.strip().lower()
        return bool(
            self._REFERENCE_PATTERN.search(normalized_question)
            or normalized_question.startswith(self._FOLLOW_UP_PREFIXES)
        )

    def set_state(self, key: str, value: object) -> None:
        """Store one session-scoped value."""

        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("state key must not be empty")
        self._state[normalized_key] = value

    def get_state(
        self,
        key: str,
        default: StateValue | None = None,
    ) -> object | StateValue | None:
        """Return one session value or a caller-provided default."""

        return self._state.get(key, default)

    def pop_state(
        self,
        key: str,
        default: StateValue | None = None,
    ) -> object | StateValue | None:
        """Remove and return one session value."""

        return self._state.pop(key, default)

    def clear(self) -> None:
        """Remove all retained turns and session state."""

        self._turns.clear()
        self._state.clear()

    def __len__(self) -> int:
        return len(self._turns)
