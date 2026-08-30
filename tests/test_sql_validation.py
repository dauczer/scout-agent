"""Security and row-boundary tests for generated SQL."""
import pytest

from agent.sql_validation import (
    MAX_RESULT_ROWS,
    SQLValidationError,
    validate_and_prepare_sql,
)


def test_missing_limit_is_added():
    sql = validate_and_prepare_sql(
        "SELECT name FROM players ORDER BY composite_score DESC"
    )
    assert f"LIMIT {MAX_RESULT_ROWS}" in sql


def test_requested_small_limit_is_preserved():
    sql = validate_and_prepare_sql("SELECT name FROM players LIMIT 3")
    assert "LIMIT 3" in sql


def test_large_limit_is_clamped():
    sql = validate_and_prepare_sql("SELECT name FROM players LIMIT 1000")
    assert f"LIMIT {MAX_RESULT_ROWS}" in sql
    assert "1000" not in sql


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE players",
        "SELECT name FROM players; DROP TABLE players",
        "PRAGMA table_info(players)",
        "ATTACH DATABASE ':memory:' AS scratch",
        "SELECT sprint_speed FROM players",
        "SELECT * FROM secret_players",
        "SELECT secret.name FROM players",
        (
            "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM x) "
            "SELECT n FROM x"
        ),
    ],
)
def test_unsafe_or_unknown_sql_is_rejected(sql):
    with pytest.raises(SQLValidationError):
        validate_and_prepare_sql(sql)


def test_cte_and_join_are_allowed():
    sql = validate_and_prepare_sql(
        """
        WITH weakest AS (
            SELECT position
            FROM club_profiles
            WHERE club_name = 'Nantes'
            ORDER BY composite_gap ASC
            LIMIT 1
        )
        SELECT p.name, p.position
        FROM players AS p
        JOIN weakest AS w ON p.position = w.position
        ORDER BY p.composite_score DESC
        LIMIT 3
        """
    )
    assert "WITH weakest AS" in sql
    assert "LIMIT 3" in sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM players WHERE preferred_foot = 'L'",
        "SELECT name FROM players WHERE position = 'MID'",
        "SELECT name FROM players WHERE position IN ('MF', 'MID')",
        "SELECT name FROM players WHERE league = 'Spain'",
    ],
)
def test_noncanonical_enum_literals_are_rejected(sql):
    with pytest.raises(SQLValidationError, match="invalid value"):
        validate_and_prepare_sql(sql)


def test_canonical_enum_literals_are_allowed():
    sql = validate_and_prepare_sql(
        "SELECT name FROM players "
        "WHERE preferred_foot = 'Left' AND position = 'MF' AND league = 'La Liga'"
    )
    assert "preferred_foot = 'Left'" in sql
