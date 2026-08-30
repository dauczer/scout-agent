"""Low-cost text-to-SQL pipeline for the public scouting demo.

The LLM produces one structured query plan. SQL validation, execution, row
limits, and table responses are deterministic and local. A second LLM call is
used only when a narrative answer materially improves the result.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time
import unicodedata
from functools import lru_cache

from groq import BadRequestError, Groq, UnprocessableEntityError
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from agent.schemas import NarrativeAnswer, QueryPlan
from agent.sql_validation import (
    MAX_RESULT_ROWS,
    SQLValidationError,
    validate_and_prepare_sql,
)
from config import settings

logger = logging.getLogger(__name__)

QUERY_TIMEOUT_SECONDS = 2.0

SYSTEM_PROMPT = f"""You convert football scouting questions into one safe SQLite query.
The database contains 2024-25 data (season '{settings.season}') for Europe's Big 5 leagues.

TABLES
players: name, team, league, season, position, age, nationality, minutes_played,
matches_played, goals_p90, assists_p90, xg_p90, xa_p90,
progressive_carries_p90, progressive_passes_p90, dribbles_completed_p90,
tackles_p90, interceptions_p90, pass_completion_pct, shot_on_target_pct,
composite_score, market_value_eur, preferred_foot, height_cm.
club_profiles: club_name, league, season, position, composite_score_avg,
league_composite_avg, composite_gap, league_rank, total_clubs.
league_averages: league, season, position, player_count, averages and standard
deviations for every player stat, plus composite_score_avg and composite_score_std.

OUTPUT DECISION
- status='query', operation='custom_sql': provide exactly one read-only SQLite SELECT.
- status='query', operation='club_weakness': use when asked which position a named club needs
  to reinforce or which group is weakest. Put the user's club name in club_name and sql=null;
  local code resolves the canonical name and computes the answer. Use this only for diagnosis;
  a question that also asks for player recommendations requires operation='custom_sql'.
- status='unsupported': the requested information is absent; sql=null and explain why.
- status='clarify': a material constraint is ambiguous; sql=null and ask one short question.
- answer_type='table' for player lists and rankings; 'text' for club diagnostics or comparisons.
- operation, sql, club_name, and message must always be present; use null where applicable.

QUERY RULES
- Use only the three listed tables and their listed columns. Never invent data.
- Produce raw SQL only in the sql field: no markdown fences or commentary.
- Add LIMIT 20 or less. Never use PRAGMA, ATTACH, DML, DDL, or multiple statements.
- Player searches: minutes_played >= 450 and ORDER BY composite_score DESC unless asked otherwise.
- Positions are exactly GK, DF, MF, FW. Map role names to the closest group, but clarify that
  centre-back/fullback and attacking/defensive midfield are not distinct database positions.
- preferred_foot values are exactly 'Left', 'Right', and 'Both'.
- league values are exactly 'Premier League', 'La Liga', 'Bundesliga', 'Serie A', and 'Ligue 1'.
- Apply every explicit age, budget, league, foot, position, and count constraint.
- Budget filters require market_value_eur IS NOT NULL. 15M means 15000000; 500k means 500000.
- 'young', 'affordable', or similar terms without a threshold are ambiguous: status='clarify'.
- If a metric is missing (for example sprint speed or goalkeeper saves), status='unsupported'.
- Club needs: use club_profiles ordered by composite_gap ASC. GK scoring measures distribution
  only, so prefer the weakest outfield group for recommendations. A CTE may combine the weakest
  group with up to three players. Exclude players already at the club being discussed.
- The configured default club name as stored in the database is '{settings.club_name}'.
"""

NARRATIVE_PROMPT = """Write a concise football scouting answer using only the supplied rows.
Do not invent facts or players. Mention important data limitations. Return 1 short paragraph or
up to 3 bullets, plus a one-sentence summary."""


class GeneratedQueryError(RuntimeError):
    """Raised after the LLM fails twice to produce executable, policy-compliant SQL."""


class SQLExecutionError(RuntimeError):
    """Raised when validated SQL cannot be executed safely."""


def _build_client() -> Groq:
    return Groq(
        api_key=settings.groq_api_key,
        timeout=12.0,
        max_retries=0,
    )


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def _structured_format(name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _generate_plan(question: str, feedback: str | None = None) -> QueryPlan:
    user_prompt = f"Question: {question}"
    if feedback:
        user_prompt += (
            "\nYour previous SQL was rejected by the local validator: "
            f"{feedback}\nReturn one corrected plan."
        )

    try:
        response = _get_client().chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=_structured_format("query_plan", QueryPlan.model_json_schema()),
            reasoning_effort="low",
            temperature=0,
            max_completion_tokens=900,
        )
    except (BadRequestError, UnprocessableEntityError) as exc:
        raise GeneratedQueryError("the provider rejected the structured query plan") from exc
    content = response.choices[0].message.content
    if not content:
        raise GeneratedQueryError("the model returned an empty query plan")
    try:
        return QueryPlan.model_validate_json(content)
    except ValidationError as exc:
        raise GeneratedQueryError("the model returned an invalid query plan") from exc


def _execute_sql(sql: str) -> list[dict]:
    from database.connection import readonly_engine

    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
    with readonly_engine.connect() as connection:
        driver_connection = connection.connection.driver_connection
        driver_connection.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0,
            1_000,
        )
        try:
            result = connection.execute(text(sql))
            rows = [dict(row) for row in result.mappings().fetchmany(MAX_RESULT_ROWS + 1)]
        except SQLAlchemyError as exc:
            raise SQLExecutionError(str(exc)) from exc
        finally:
            driver_connection.set_progress_handler(None, 0)

    if len(rows) > MAX_RESULT_ROWS:
        raise SQLExecutionError("query returned more rows than the enforced limit")
    return rows


def _table_summary(rows: list[dict]) -> str:
    count = len(rows)
    if count == 0:
        return "No matching players or scouting records were found."
    names = [str(row["name"]) for row in rows if row.get("name")][:3]
    if names:
        return f"Found {count} result{'s' if count != 1 else ''}: {', '.join(names)}."
    return f"Found {count} scouting record{'s' if count != 1 else ''}."


@lru_cache(maxsize=1)
def _known_club_names() -> tuple[str, ...]:
    from database.connection import readonly_engine

    with readonly_engine.connect() as connection:
        result = connection.execute(
            text("SELECT DISTINCT club_name FROM club_profiles ORDER BY club_name")
        )
        return tuple(str(value) for value in result.scalars() if value)


def _club_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    words = re.findall(r"[a-z0-9]+", normalized)
    removable = {"ac", "afc", "cf", "fc", "ssc"}
    while words and words[0] in removable:
        words.pop(0)
    while words and words[-1] in removable:
        words.pop()
    return "".join(words)


def _resolve_club_name(requested: str) -> str | None:
    clubs = _known_club_names()
    requested_key = _club_key(requested)
    aliases = {
        "parissaintgermain": "Paris S-G",
        "psg": "Paris S-G",
    }
    if requested_key in aliases and aliases[requested_key] in clubs:
        return aliases[requested_key]

    exact_matches = [club for club in clubs if _club_key(club) == requested_key]
    if len(exact_matches) == 1:
        return exact_matches[0]

    partial_matches = [
        club
        for club in clubs
        if requested_key and (
            requested_key in _club_key(club) or _club_key(club) in requested_key
        )
    ]
    return partial_matches[0] if len(partial_matches) == 1 else None


def _club_weakness_response(requested_club: str) -> dict:
    club_name = _resolve_club_name(requested_club)
    if club_name is None:
        message = (
            f"I could not match '{requested_club}' to exactly one club in the dataset. "
            "Please use a more specific club name."
        )
        return {
            "type": "text",
            "data": [{"text": message}],
            "summary": message,
            "sql": None,
        }

    safe_club = club_name.replace("'", "''")
    safe_season = settings.season.replace("'", "''")
    sql = validate_and_prepare_sql(
        "SELECT position, composite_score_avg, league_composite_avg, "
        "composite_gap, league_rank, total_clubs FROM club_profiles "
        f"WHERE club_name = '{safe_club}' AND season = '{safe_season}' "
        "AND composite_gap IS NOT NULL "
        "ORDER BY composite_gap ASC LIMIT 4"
    )
    rows = _execute_sql(sql)
    outfield_rows = [row for row in rows if row.get("position") != "GK"]
    if not outfield_rows:
        message = f"No outfield profile is available for {club_name}."
        return {
            "type": "text",
            "data": [{"text": message}],
            "summary": message,
            "sql": sql,
        }

    weakest = outfield_rows[0]
    labels = {"DF": "defence", "MF": "midfield", "FW": "attack"}
    position = str(weakest["position"])
    label = labels.get(position, position)
    gap = weakest.get("composite_gap")
    rank = weakest.get("league_rank")
    total = weakest.get("total_clubs")
    gap_text = f"{float(gap):+.2f}" if isinstance(gap, int | float) else "not available"
    rank_text = f"{rank}/{total}" if rank is not None and total is not None else "not available"
    text_answer = (
        f"For {club_name}, the weakest measured outfield group is {label} ({position}), "
        f"with a composite gap of {gap_text} versus its league average and a "
        f"league rank of {rank_text}. Goalkeepers are excluded from this "
        "reinforcement priority because their score only measures distribution."
    )
    summary = f"Priority for {club_name}: {label} ({position})."
    return {
        "type": "text",
        "data": [{"text": text_answer}],
        "summary": summary,
        "sql": sql,
    }


def _narrate(question: str, rows: list[dict]) -> NarrativeAnswer:
    response = _get_client().chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": NARRATIVE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nDatabase rows: "
                    f"{json.dumps(rows, ensure_ascii=False, default=str)}"
                ),
            },
        ],
        response_format=_structured_format(
            "narrative_answer",
            NarrativeAnswer.model_json_schema(),
        ),
        reasoning_effort="low",
        temperature=0,
        max_completion_tokens=700,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("the model returned an empty narrative")
    return NarrativeAnswer.model_validate_json(content)


def _non_query_response(plan: QueryPlan) -> dict:
    message = plan.message or "This question cannot be answered from the available data."
    return {
        "type": "text",
        "data": [{"text": message}],
        "summary": message,
        "sql": None,
    }


@lru_cache(maxsize=256)
def _scout_query_cached(question: str) -> dict:
    feedback: str | None = None
    plan: QueryPlan | None = None
    validated_sql: str | None = None
    rows: list[dict] | None = None

    for attempt in range(2):
        plan = _generate_plan(question, feedback)
        if plan.status != "query":
            return _non_query_response(plan)
        if plan.operation == "club_weakness":
            return _club_weakness_response(plan.club_name or "")

        try:
            validated_sql = validate_and_prepare_sql(plan.sql or "")
            rows = _execute_sql(validated_sql)
            break
        except (SQLValidationError, SQLExecutionError) as exc:
            if attempt == 0:
                feedback = str(exc)[:500]
                continue
            raise GeneratedQueryError(
                f"generated SQL remained invalid after one retry: {exc}"
            ) from exc

    if plan is None or validated_sql is None or rows is None:
        raise GeneratedQueryError("the query plan did not produce a result")

    if plan.answer_type == "text":
        if not rows:
            return {
                "type": "text",
                "data": [{"text": "No matching data was found for this question."}],
                "summary": "No matching scouting data was found.",
                "sql": validated_sql,
            }
        try:
            narrative = _narrate(question, rows)
            return {
                "type": "text",
                "data": [{"text": narrative.text}],
                "summary": narrative.summary,
                "sql": validated_sql,
            }
        except Exception:
            logger.exception("narrative generation failed; returning deterministic table fallback")

    return {
        "type": "table",
        "data": rows,
        "summary": _table_summary(rows),
        "sql": validated_sql,
    }


def scout_query(question: str) -> dict:
    """Run a normalized, cached scouting query."""
    normalized = " ".join(question.split())
    return copy.deepcopy(_scout_query_cached(normalized))


def clear_scout_cache() -> None:
    """Clear cached scouting responses (primarily useful for tests and reseeds)."""
    _scout_query_cached.cache_clear()
