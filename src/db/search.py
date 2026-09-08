"""
search.py

Provides AI-friendly search over a loaded database schema.


Author: Jianhua Shao
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .healthcare_terms import HealthcareTerms
from .synonyms import SynonymManager
from .types import DatabaseSchema
from .catalog import Catalog

ValueCatalogView = Mapping[str, Mapping[str, Sequence[object]]]


class SchemaSearch:
    """Search schema metadata using vocabulary, concepts, codes, and values."""

    def __init__(
        self,
        catalog: Catalog,
        synonyms: SynonymManager | None = None,
        healthcare_terms: HealthcareTerms | None = None,
    ) -> None:
        self._tables = MappingProxyType({
            table: MappingProxyType(dict(entry))
            for table, entry in catalog['tables'].items()
        })
        self._schema: DatabaseSchema = MappingProxyType({
            table: MappingProxyType({
                column: MappingProxyType(dict(info))
                for column, info in columns.items()
            })
            for table, columns in catalog['columns'].items()
        })
        self._values: ValueCatalogView = MappingProxyType({
            table: MappingProxyType({
                column: tuple(values)
                for column, values in columns.items()
            })
            for table, columns in catalog['values'].items()
        })
        self._relationships = tuple(
            MappingProxyType(dict(relationship))
            for relationship in catalog['relationships']
        )
        self._synonyms = synonyms or SynonymManager()
        self._healthcare_terms = healthcare_terms or HealthcareTerms()

    def expand_query(self, text: str) -> set[str]:
        """Return lexical and clinical terms used by the search pipeline."""
        terms = self._synonyms.expand(text)
        terms.update(self._healthcare_terms.expand(text))
        return {term.lower().strip() for term in terms if term.strip()}

    def find_tables(self, text: str) -> list[str]:
        """Return tables whose names contain a word from *text*."""
        terms = self.expand_query(text)

        matches: list[str] = []

        for table in self._tables:
            name = self._normalize_name(table)
            if self._matches_any(name, terms):
                matches.append(table)

        return sorted(matches)

    def find_columns(self, keyword: str) -> list[tuple[str, str]]:
        """Return columns related to a query, including clinical codes/values."""
        terms = self.expand_query(keyword)
        codes = self._medical_codes(keyword)

        results: list[tuple[str, str]] = []

        for table, columns in self._schema.items():
            for column in columns:
                if self._column_matches(table, column, terms, codes):
                    results.append((table, column))

        return results

    def get_related_schema(self, text: str) -> dict[str, list[str]]:
        """Return tables and columns related to words in *text*."""
        terms = self.expand_query(text)
        codes = self._medical_codes(text)
        result: dict[str, list[str]] = {}

        for table, columns in self._schema.items():
            if self._matches_any(self._normalize_name(table), terms):
                result[table] = list(columns.keys())
                continue

            matched_columns = [
                column
                for column in columns
                if self._column_matches(table, column, terms, codes)
            ]

            if matched_columns:
                result[table] = matched_columns

        return result

    def _medical_codes(self, text: str) -> set[str]:
        return {
            concept.code.lower()
            for concept in self._healthcare_terms.find_in_text(text)
        }

    def _column_matches(
        self,
        table: str,
        column: str,
        terms: set[str],
        medical_codes: set[str],
    ) -> bool:
        normalized_column = self._normalize_name(column)
        if self._matches_any(normalized_column, terms):
            return True

        values = self._values.get(table, {}).get(column, ())
        normalized_values = {str(value).lower().strip() for value in values}
        if terms.intersection(normalized_values):
            return True

        # A resolved clinical code makes generic code-bearing columns relevant,
        # even when a sampled value catalog does not contain that exact code.
        return bool(medical_codes) and 'code' in normalized_column.split()

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r'[^a-z0-9]+', ' ', name.lower()).strip()

    @staticmethod
    def _matches_any(name: str, terms: set[str]) -> bool:
        return any(
            SchemaSearch._normalize_name(term) in name
            for term in terms
        )
