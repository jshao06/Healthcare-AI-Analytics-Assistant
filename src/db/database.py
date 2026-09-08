"""Database interface shared by database-specific adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd


class Database(ABC):
    """Interface required by schema and catalog components."""

    @abstractmethod
    def query(
        self,
        sql: str,
        parameters: Sequence[object] = (),
        *,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Execute SQL with bound parameters and an optional row cap."""

    @abstractmethod
    def list_tables(self) -> list[str]:
        """Return the available table names."""

    @abstractmethod
    def table_info(self, table_name: str) -> pd.DataFrame:
        """Return metadata for the columns in a table."""

    @abstractmethod
    def quote_identifier(self, identifier: str) -> str:
        """Quote an identifier according to the database dialect."""
