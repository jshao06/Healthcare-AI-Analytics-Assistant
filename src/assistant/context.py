"""Build relevant database context for a user question."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..db.catalog import CatalogBuilder
from ..db.database import Database
from ..db.healthcare_terms import HealthcareTerms
from ..db.schema import SchemaManager
from ..db.search import SchemaSearch
from ..db.sqlite_database import SQLiteDatabase
from ..db.synonyms import SynonymManager


@dataclass(frozen=True, slots=True)
class ColumnContext:
    """Basic metadata for one relevant column."""

    name: str
    data_type: str
    nullable: bool
    primary_key: bool


@dataclass(frozen=True, slots=True)
class TableContext:
    """A relevant table and its matching columns."""

    name: str
    columns: tuple[ColumnContext, ...]


@dataclass(frozen=True, slots=True)
class Context:
    """Database context relevant to a user question."""

    question: str
    relevant_tables: tuple[TableContext, ...]
    relationships: tuple[str, ...] = ()


class ContextBuilder:
    """Find database schema relevant to a user question."""

    def __init__(
        self,
        database: Database,
        *,
        schema_manager: SchemaManager | None = None,
        synonyms: SynonymManager | None = None,
        healthcare_terms: HealthcareTerms | None = None,
    ) -> None:
        self._schema_manager = schema_manager or SchemaManager(database)

        catalog = CatalogBuilder(
            database,
            self._schema_manager.get_schema(),
        ).build()

        self._catalog = catalog
        self._search = SchemaSearch(
            catalog,
            synonyms=synonyms or SynonymManager(),
            healthcare_terms=healthcare_terms or HealthcareTerms(),
        )

    @classmethod
    def from_sqlite(
        cls,
        database_path: str | Path,
    ) -> ContextBuilder:
        """Create a context builder for a SQLite database."""

        return cls(SQLiteDatabase(database_path))

    def build(self, question: str) -> Context:
        """Build database context for the supplied question."""

        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        related_schema = self._search.get_related_schema(question)

        return Context(
            question=question,
            relevant_tables=self._build_tables(related_schema),
            relationships=self._build_relationships(related_schema),
        )

    def _build_tables(
        self,
        related_schema: dict[str, list[str]],
    ) -> tuple[TableContext, ...]:
        """Build table and column context from related schema."""

        schema = self._schema_manager.get_schema()
        tables: list[TableContext] = []

        for table_name, column_names in sorted(related_schema.items()):
            columns = tuple(
                ColumnContext(
                    name=column_name,
                    data_type=str(schema[table_name][column_name]["type"]),
                    nullable=bool(
                        schema[table_name][column_name]["nullable"]
                    ),
                    primary_key=bool(
                        schema[table_name][column_name]["primary_key"]
                    ),
                )
                for column_name in column_names
            )

            tables.append(
                TableContext(
                    name=table_name,
                    columns=columns,
                )
            )

        return tuple(tables)

    def _build_relationships(
        self,
        related_schema: dict[str, list[str]],
    ) -> tuple[str, ...]:
        """Return relationships involving relevant tables."""

        relevant_tables = set(related_schema)

        relationships = {
            (
                f"{relationship['from_table']}."
                f"{relationship['from_column']} -> "
                f"{relationship['to_table']}."
                f"{relationship['to_column']}"
            )
            for relationship in self._catalog["relationships"]
            if relationship["from_table"] in relevant_tables
            or relationship["to_table"] in relevant_tables
        }

        return tuple(sorted(relationships))


__all__ = [
    "ColumnContext",
    "Context",
    "ContextBuilder",
    "TableContext",
]