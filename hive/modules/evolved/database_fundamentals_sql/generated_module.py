import sqlite3
from typing import List, Optional, Tuple

import pytest


class DatabaseFundamentalsSqlManager:
    """A class to demonstrate database fundamentals in SQL operations."""

    def __init__(self, db_path: str):
        """Initialize the database connection."""
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

    def create_table(self, table_name: str, columns: List[Tuple[str, str]]) -> None:
        """
        Create a table in the database with the given name and columns.

        :param table_name: The name of the table to create.
        :param columns: A list of tuples, each containing the column name and data type.
        """
        column_definitions = ", ".join(
            [f"{col_name} {data_type}" for col_name, data_type in columns]
        )
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})"
        self.cursor.execute(create_table_query)

    def insert_data(self, table_name: str, data: List[Tuple]) -> None:
        """
        Insert data into a table in the database.

        :param table_name: The name of the table to insert data into.
        :param data: A list of tuples, each containing the data to insert.
        """
        placeholders = ", ".join(["?" for _ in data[0]])
        insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"
        self.cursor.executemany(insert_query, data)

    def select_data(self, table_name: str, condition: Optional[str] = None) -> List[Tuple]:
        """
        Select data from a table in the database, optionally with a condition.

        :param table_name: The name of the table to select data from.
        :param condition: An optional SQL WHERE clause condition.
        :return: A list of tuples containing the selected data.
        """
        select_query = f"SELECT * FROM {table_name}"
        if condition:
            select_query += f" WHERE {condition}"
        self.cursor.execute(select_query)
        return self.cursor.fetchall()

    def close_connection(self) -> None:
        """Close the database connection."""
        self.connection.close()


# Unit tests for the DatabaseFundamentalsSqlManager class
def test_database_fundamentals_sql():
    test_db_path = ":memory:"
    db_manager = DatabaseFundamentalsSqlManager(test_db_path)

    # Test create_table method
    db_manager.create_table(
        "users", [("id", "INTEGER PRIMARY KEY"), ("name", "TEXT"), ("age", "INTEGER")]
    )
    assert db_manager.select_data("users", condition="1=1") == []

    # Test insert_data method
    db_manager.insert_data("users", [(1, "Alice", 30), (2, "Bob", 25)])
    assert db_manager.select_data("users", condition="1=1") == [(1, "Alice", 30), (2, "Bob", 25)]

    # Test select_data method
    assert db_manager.select_data("users", condition="age > 25") == [(1, "Alice", 30)]

    # Test close_connection method
    db_manager.close_connection()
    with pytest.raises(sqlite3.OperationalError):
        db_manager.select_data("users", condition="1=1")


# Run the unit tests
if __name__ == "__main__":
    test_database_fundamentals_sql()
