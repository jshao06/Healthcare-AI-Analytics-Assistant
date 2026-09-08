"""
schema.py
Loads and manages the SQLite database schema.

The SchemaManager:
1) caches metadata about every table and column,
2) allows other modules (SQL generator, validator, AI agent, etc.) to access schema information without repeatedly querying SQLite.

Author: Jianhua Shao
"""

from __future__ import annotations
from types import MappingProxyType
from typing import cast

from .database import Database
from .types import (
    ColumnInfoView,
    DatabaseSchema,
    MutableDatabaseSchema,
    MutableTableSchema,
)

class SchemaManager:
    """
    Manages the database schema.

    Parameters
    ----------
    database: Database
        Database helper object.
    """
    
    def __init__(self, database: Database) -> None:
        self.database = database    
        self._schema: MutableDatabaseSchema = {}
        self.load_schema()
        
    # ---------------------------------------------------------
    # Loading
    # ---------------------------------------------------------

    def load_schema(self) -> None:
        """
        Load all tables and columns from the database.
        """
        
        self._schema.clear()

        for table in self.database.list_tables():
            info = self.database.table_info(table)
            records = cast(
                list[dict[str, object]],
                info.to_dict(orient='records'),
            )
            columns: MutableTableSchema = {}

            for record in records:
                column_name = str(record['name'])
                column_type = str(record['type'])
                is_not_null = bool(record['notnull'])
                is_primary_key = bool(record['pk'])
                default_value = record.get('dflt_value')

                columns[column_name] = {
                    'type': column_type,
                    'nullable': not is_not_null,
                    'primary_key': is_primary_key,
                    'default': default_value,
                }

            self._schema[table] = columns

            
    def reload(self) -> None:
        """
        Reload schema from the database
        """
        self.load_schema()

    # ---------------------------------------------------------
    # Getters
    # ---------------------------------------------------------
    
    def get_schema(self) -> DatabaseSchema:
        """
        Return a recursively read-only view of the complete schema.
        """
        tables = {
            table: MappingProxyType({
                column: MappingProxyType(info)
                for column, info in columns.items()
            })
            for table, columns in self._schema.items()
        }
        return MappingProxyType(tables)
        
    def get_table_names(self) -> list[str]:
        """
        Return all table names.
        """
        return list(self._schema.keys())

    def has_table(self, table_name: str) -> bool:
        """
        Check whether a table exists.
        """
        return table_name in self._schema

    def get_columns(self, table_name: str) -> list[str]:
        """
        Return column names for a table.
        """
        if not self.has_table(table_name):
            raise ValueError(f'Unknown table: {table_name}')

        return list(self._schema[table_name].keys())

    def get_primary_key(self, table_name: str) -> str | None:
        """Return the primary-key column for a table, if one exists."""

        if not self.has_table(table_name):
            raise ValueError(f'Unknown table: {table_name}')

        return next(
            (
                column
                for column, info in self._schema[table_name].items()
                if info['primary_key']
            ),
            None,
        )
        
    def has_column(self, table_name: str, column_name: str) -> bool:
        """
        Check whether a column exists.
        """
        if not self.has_table(table_name):
            return False

        return column_name in self._schema[table_name]
    
    def get_column_info(self, table_name: str, column_name: str) -> ColumnInfoView:
        """
        Return metadata for a column.
        """
        if not self.has_column(table_name, column_name):
            raise ValueError(f'{table_name}.{column_name} does not exist.')
            
        return MappingProxyType(self._schema[table_name][column_name])
 
    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------
       
    def schema_text(self) -> str:
        """
        Return a human-readable schema for LLM prompts.
        """
        lines: list[str] = []

        for table in self.get_table_names():

            lines.append(f'Table: {table}')
            lines.append('-' * 50)

            for column, info in self._schema[table].items():

                line = (f'{column}' f' ({info["type"]})')

                if info['primary_key']:
                    line += ' [PK]'

                if not info['nullable']:
                    line += ' NOT NULL'

                lines.append(line)

            lines.append('')

        return '\n'.join(lines)

    def __str__(self):
        return self.schema_text()
