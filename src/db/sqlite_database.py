"""
sqlite_database.py
Provides a simple interface for connecting to the TinyEHR SQLite database and executing SQL queries.
Author: Jianhua Shao
"""

from pathlib import Path
import sqlite3 as sql3
from collections.abc import Sequence
from contextlib import closing
from typing import Any, cast

import pandas as pd

from .database import Database

class SQLiteDatabase(Database):
    
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f'Database not found: {self.db_path}')

    def query(
        self,
        sql: str,
        parameters: Sequence[object] = (),
        *,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """
        The connection is opened using a context manager.
        Then execute a SQL query and return a DataFrame.
        """
        query_sql = sql
        query_parameters = tuple(parameters)

        if max_rows is not None:
            if max_rows < 1:
                raise ValueError('max_rows must be at least 1')
            inner_sql = sql.strip().removesuffix(';')
            query_sql = (
                'SELECT * FROM ('
                f'{inner_sql}'
                ') AS "_limited_query" LIMIT ?'
            )
            query_parameters += (max_rows,)

        pandas_parameters = cast(
            list[Any],
            list(query_parameters),
        )

        with closing(sql3.connect(self.db_path)) as conn:
            return pd.read_sql_query(  # pyright: ignore[reportUnknownMemberType]
                query_sql,
                conn,
                params=pandas_parameters,
            )

    def list_tables(self) -> list[str]:
        """
        Return a list of all table names sorted. 
        """
        sql = 'select name from sqlite_master where type="table" order by name'
        df = self.query(sql)
        return cast(list[str], df['name'].tolist())

    def table_info(self, table_name: str) -> pd.DataFrame:
        """
        Return table columns as DataFrame.
        """ 
        sql = f'pragma table_info({self.quote_identifier(table_name)})'
        return self.query(sql)

    def quote_identifier(self, identifier: str) -> str:
        """Quote an SQLite identifier and escape embedded double quotes."""
        return '"' + identifier.replace('"', '""') + '"'
        
