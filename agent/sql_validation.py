"""Deterministic validation and row limiting for generated SQLite queries."""
from __future__ import annotations

from sqlglot import exp, parse
from sqlglot.errors import ParseError

MAX_RESULT_ROWS = 20

ALLOWED_COLUMNS: dict[str, set[str]] = {
    "players": {
        "id", "name", "team", "league", "season", "nationality", "position",
        "age", "preferred_foot", "height_cm", "market_value_eur", "minutes_played",
        "matches_played", "goals_p90", "assists_p90", "xg_p90", "xa_p90",
        "progressive_carries_p90", "progressive_passes_p90",
        "dribbles_completed_p90", "tackles_p90", "interceptions_p90",
        "pass_completion_pct", "shot_on_target_pct", "goals_p90_zscore",
        "assists_p90_zscore", "xg_p90_zscore", "xa_p90_zscore",
        "progressive_carries_p90_zscore", "progressive_passes_p90_zscore",
        "dribbles_completed_p90_zscore", "tackles_p90_zscore",
        "interceptions_p90_zscore", "pass_completion_pct_zscore",
        "shot_on_target_pct_zscore", "composite_score",
    },
    "club_profiles": {
        "id", "club_name", "league", "season", "position", "composite_score_avg",
        "league_composite_avg", "composite_gap", "league_rank", "total_clubs",
    },
    "league_averages": {
        "id", "league", "season", "position", "player_count", "goals_p90_avg",
        "goals_p90_std", "assists_p90_avg", "assists_p90_std", "xg_p90_avg",
        "xg_p90_std", "xa_p90_avg", "xa_p90_std",
        "progressive_carries_p90_avg", "progressive_carries_p90_std",
        "progressive_passes_p90_avg", "progressive_passes_p90_std",
        "dribbles_completed_p90_avg", "dribbles_completed_p90_std",
        "tackles_p90_avg", "tackles_p90_std", "interceptions_p90_avg",
        "interceptions_p90_std", "pass_completion_pct_avg",
        "pass_completion_pct_std", "shot_on_target_pct_avg",
        "shot_on_target_pct_std", "composite_score_avg", "composite_score_std",
    },
}

_FORBIDDEN_NODE_NAMES = {
    "Alter", "Analyze", "Attach", "Command", "Commit", "Copy", "Create",
    "Delete", "Detach", "Drop", "Execute", "Grant", "Insert", "LoadData",
    "Merge", "Pragma", "Rollback", "Set", "Transaction", "TruncateTable",
    "Update", "Use",
}

_ENUM_VALUES: dict[str, set[str]] = {
    "preferred_foot": {"Both", "Left", "Right"},
    "position": {"DF", "FW", "GK", "MF"},
    "league": {"Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"},
}


class SQLValidationError(ValueError):
    """Raised when generated SQL is outside the portfolio demo's read-only policy."""


def _normalise_name(value: str | None) -> str:
    return (value or "").casefold()


def _validate_limit(statement: exp.Query) -> exp.Query:
    limit = statement.args.get("limit")
    if limit is None:
        return statement.limit(MAX_RESULT_ROWS, copy=False)

    expression = limit.args.get("expression")
    if not isinstance(expression, exp.Literal) or not expression.is_int:
        raise SQLValidationError("LIMIT must be a positive integer")

    value = int(expression.this)
    if value <= 0:
        raise SQLValidationError("LIMIT must be a positive integer")
    if value > MAX_RESULT_ROWS:
        return statement.limit(MAX_RESULT_ROWS, copy=False)
    return statement


def _validate_enum_literal(column: exp.Expression, literal: exp.Expression) -> None:
    if not isinstance(column, exp.Column) or not isinstance(literal, exp.Literal):
        return
    if not literal.is_string:
        return

    column_name = _normalise_name(column.name)
    allowed = _ENUM_VALUES.get(column_name)
    if allowed is not None and literal.this not in allowed:
        choices = ", ".join(sorted(allowed))
        raise SQLValidationError(
            f"invalid value '{literal.this}' for {column.name}; use one of: {choices}"
        )


def _validate_enum_literals(statement: exp.Query) -> None:
    """Reject non-canonical enum filters so the model can correct them once."""
    for equality in statement.find_all(exp.EQ):
        _validate_enum_literal(equality.this, equality.expression)
        _validate_enum_literal(equality.expression, equality.this)

    for membership in statement.find_all(exp.In):
        column = membership.this
        for literal in membership.expressions:
            _validate_enum_literal(column, literal)


def validate_and_prepare_sql(sql: str) -> str:
    """Return canonical SQLite SQL after enforcing a single, bounded read query."""
    if not sql or not sql.strip():
        raise SQLValidationError("SQL is empty")

    try:
        statements = [statement for statement in parse(sql, read="sqlite") if statement]
    except ParseError as exc:
        raise SQLValidationError(f"SQL could not be parsed: {exc}") from exc

    if len(statements) != 1:
        raise SQLValidationError("exactly one SQL statement is allowed")

    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise SQLValidationError("only SELECT queries are allowed")

    for node in statement.walk():
        if type(node).__name__ in _FORBIDDEN_NODE_NAMES:
            raise SQLValidationError(f"{type(node).__name__.upper()} is not allowed")

    with_clause = statement.args.get("with") or statement.args.get("with_")
    if with_clause is not None and bool(with_clause.args.get("recursive")):
        raise SQLValidationError("recursive CTEs are not allowed")

    cte_names = {
        _normalise_name(cte.alias_or_name)
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    }
    derived_sources = {
        _normalise_name(subquery.alias_or_name)
        for subquery in statement.find_all(exp.Subquery)
        if subquery.alias_or_name
    } | cte_names

    table_aliases: dict[str, str | None] = {}
    real_tables: set[str] = set()
    for table in statement.find_all(exp.Table):
        table_name = _normalise_name(table.name)
        alias = _normalise_name(table.alias_or_name) or table_name
        if table_name in cte_names:
            table_aliases[alias] = None
            continue
        if table_name not in ALLOWED_COLUMNS:
            raise SQLValidationError(f"table '{table.name}' is not allowed")
        real_tables.add(table_name)
        table_aliases[alias] = table_name
        table_aliases[table_name] = table_name

    if not real_tables:
        raise SQLValidationError("query must read from an allowed scouting table")

    derived_columns = {
        _normalise_name(alias.alias)
        for alias in statement.find_all(exp.Alias)
        if alias.alias
    }
    known_columns = set().union(*(ALLOWED_COLUMNS[table] for table in real_tables))

    for column in statement.find_all(exp.Column):
        column_name = _normalise_name(column.name)
        if not column_name or column_name == "*":
            continue

        qualifier = _normalise_name(column.table)
        if qualifier:
            if qualifier in derived_sources:
                continue
            if qualifier not in table_aliases:
                raise SQLValidationError(f"unknown table alias '{column.table}'")
            source_table = table_aliases.get(qualifier)
            if source_table is None:
                continue
            if column_name not in ALLOWED_COLUMNS[source_table]:
                raise SQLValidationError(
                    f"column '{column.name}' does not exist on table '{source_table}'"
                )
        elif column_name not in known_columns and column_name not in derived_columns:
            raise SQLValidationError(f"column '{column.name}' is not allowed")

    _validate_enum_literals(statement)
    statement = _validate_limit(statement)
    return statement.sql(dialect="sqlite")
