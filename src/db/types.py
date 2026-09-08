from collections.abc import Mapping
from typing import TypedDict

class ColumnInfo(TypedDict):
    type: str
    nullable: bool
    primary_key: bool
    default: object | None

MutableTableSchema = dict[str, ColumnInfo]
MutableDatabaseSchema = dict[str, MutableTableSchema]

ColumnInfoView = Mapping[str, object]
TableSchema = Mapping[str, ColumnInfoView]
DatabaseSchema = Mapping[str, TableSchema]
