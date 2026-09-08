"""
catalog.py

Builds searchable metadata catalogs for SQL generation.

Author: Jianhua Shao
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, TypeAlias, TypedDict, cast

from .database import Database
from .types import DatabaseSchema


class TableCatalogEntry(TypedDict):
    """Metadata recorded for a database table."""

    column_count: int


class Relationship(TypedDict):
    """A joinable relationship between two schema columns."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str
    source: Literal['declared', 'inferred']


TableCatalog: TypeAlias = dict[str, TableCatalogEntry]
ValueCatalog: TypeAlias = dict[str, dict[str, list[object]]]
RelationshipKey: TypeAlias = tuple[str, str, str, str]


class Catalog(TypedDict):
    """The four metadata catalogs used by SQL generation."""

    tables: TableCatalog
    columns: DatabaseSchema
    relationships: list[Relationship]
    values: ValueCatalog

# TinyEHR/MIMIC-style tables do not declare SQLite foreign keys.  These stable
# keys provide the join targets that a SQL generation agent needs.
INFERRED_KEY_TARGETS: dict[str, tuple[str, str]] = {
    'subject_id': ('patients', 'subject_id'),
    'hadm_id': ('admissions', 'hadm_id'),
    'stay_id': ('icustays', 'stay_id'),
    'caregiver_id': ('caregiver', 'caregiver_id'),
    'provider_id': ('provider', 'provider_id'),
    'poe_id': ('poe', 'poe_id'),
    'pharmacy_id': ('pharmacy', 'pharmacy_id'),
    'emar_id': ('emar', 'emar_id'),
}

# These names conventionally hold free-form content, not category labels.
UNSTRUCTURED_TEXT_NAMES: set[str] = {'comment', 'note', 'notes', 'text', 'value'}


class CatalogBuilder:
    """Build metadata catalogs from a database and its loaded schema.

    The resulting catalog has four sections: ``tables``, ``columns``,
    ``relationships``, and ``values``.
    """

    def __init__(
        self,
        database: Database,
        schema: DatabaseSchema,
        value_limit: int = 50,
    ) -> None:
        if value_limit < 1:
            raise ValueError('value_limit must be at least 1')

        self.database = database
        self.schema: DatabaseSchema = MappingProxyType({
            table: MappingProxyType({
                column: MappingProxyType(dict(info))
                for column, info in columns.items()
            })
            for table, columns in schema.items()
        })
        self.value_limit = value_limit

    def build(self) -> Catalog:
        """Build all catalogs used by the SQL generation agent."""
        return {
            'tables': self.build_table_catalog(),
            'columns': self.build_column_catalog(),
            'relationships': self.build_relationship_catalog(),
            'values': self.build_value_catalog(),
        }

    def build_table_catalog(self) -> TableCatalog:
        """Return available tables and their number of columns."""
        return {
            table: {'column_count': len(columns)}
            for table, columns in self.schema.items()
        }

    def build_column_catalog(self) -> DatabaseSchema:
        """Return column metadata keyed by table and column name."""
        return self.schema

    def build_relationship_catalog(self) -> list[Relationship]:
        """Return declared and inferred table relationships."""
        relationships: list[Relationship] = []
        seen: set[RelationshipKey] = set()

        for table in self.schema:
            pragma = f'PRAGMA foreign_key_list({self.database.quote_identifier(table)})'
            for _, row in self.database.query(pragma).iterrows():
                relationship: Relationship = {
                    'from_table': table,
                    'from_column': str(row['from']),
                    'to_table': str(row['table']),
                    'to_column': str(row['to']),
                    'source': 'declared',
                }
                relationships.append(relationship)
                seen.add(self._relationship_key(relationship))

        for table, columns in self.schema.items():
            for column in columns:
                target = self._inferred_relationship_target(column)
                if target is None:
                    continue

                target_table, target_column = target
                if table == target_table or target_column not in self.schema.get(target_table, {}):
                    continue

                relationship: Relationship = {
                    'from_table': table,
                    'from_column': column,
                    'to_table': target_table,
                    'to_column': target_column,
                    'source': 'inferred',
                }
                key = self._relationship_key(relationship)
                if key not in seen:
                    relationships.append(relationship)
                    seen.add(key)

        return relationships

    def build_value_catalog(self) -> ValueCatalog:
        """Return representative values for low-cardinality categorical columns."""
        catalog: ValueCatalog = {}

        for table, columns in self.schema.items():
            for column, info in columns.items():
                column_type = cast(str | None, info['type'])
                if not self.should_sample_values(table, column, column_type):
                    continue

                values = self._representative_values(table, column)
                if values is not None:
                    catalog.setdefault(table, {})[column] = values

        return catalog

    def should_sample_values(
        self,
        table: str,
        column: str,
        column_type: str | None = None,
    ) -> bool:
        """Return whether a column is a likely categorical text field."""
        del table
        normalized_type = (column_type or '').upper()
        is_text = any(token in normalized_type for token in ('CHAR', 'TEXT', 'CLOB'))
        is_identifier = column.lower() == 'id' or column.lower().endswith('_id')
        is_unstructured = column.lower() in UNSTRUCTURED_TEXT_NAMES
        return is_text and not is_identifier and not is_unstructured

    def _representative_values(self, table: str, column: str) -> list[object] | None:
        quoted_table = self.database.quote_identifier(table)
        quoted_column = self.database.quote_identifier(column)

        cardinality_query = (
            f'SELECT COUNT(DISTINCT {quoted_column}) AS distinct_count '
            f'FROM {quoted_table} '
            f'WHERE {quoted_column} IS NOT NULL'
        )
        cardinality_result = self.database.query(cardinality_query)
        cardinality_records = cast(
            list[dict[str, object]],
            cardinality_result.to_dict(orient='records'),
        )
        distinct_count = int(str(cardinality_records[0]['distinct_count']))

        if distinct_count > self.value_limit:
            return None

        frequency_query = (
            f'SELECT {quoted_column} '
            f'FROM {quoted_table} '
            f'WHERE {quoted_column} IS NOT NULL '
            f'GROUP BY {quoted_column} '
            f'ORDER BY COUNT(*) DESC, {quoted_column} '
            f'LIMIT {self.value_limit}'
        )
        frequency_records = cast(
            list[dict[str, object]],
            self.database.query(frequency_query).to_dict(orient='records'),
        )
        values = [record[column] for record in frequency_records]

        # Low cardinality alone is not enough for a text field to be a useful
        # categorical prompt hint. Avoid retaining prose-like values.
        if any(isinstance(value, str) and len(value) > 100 for value in values):
            return None

        return values

    @staticmethod
    def _inferred_relationship_target(column: str) -> tuple[str, str] | None:
        """Return the entity table referenced by a conventional key column."""
        for key, target in INFERRED_KEY_TARGETS.items():
            if column == key or column.endswith(f'_{key}'):
                return target
        return None

    @staticmethod
    def _relationship_key(relationship: Relationship) -> RelationshipKey:
        return (
            relationship['from_table'],
            relationship['from_column'],
            relationship['to_table'],
            relationship['to_column'],
        )
