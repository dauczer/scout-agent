"""Structured contracts for planning and returning scouting queries."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class QueryPlan(BaseModel):
    """One constrained decision produced by the LLM before local validation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["query", "unsupported", "clarify"]
    operation: Literal["custom_sql", "club_weakness"]
    answer_type: Literal["table", "text"]
    sql: str | None
    club_name: str | None
    message: str | None

    @model_validator(mode="after")
    def validate_status_fields(self) -> "QueryPlan":
        if self.status == "query" and self.operation == "custom_sql" and not self.sql:
            raise ValueError("custom SQL plans must include SQL")
        if self.status == "query" and self.operation == "club_weakness" and not self.club_name:
            raise ValueError("club weakness plans must include a club name")
        if self.status != "query" and not self.message:
            raise ValueError("non-query plans must include a user-facing message")
        return self


class NarrativeAnswer(BaseModel):
    """Optional prose rendering for club diagnostics and comparisons."""

    model_config = ConfigDict(extra="forbid")

    text: str
    summary: str


class ScoutResponse(BaseModel):
    """Stable API response, including the validated SQL shown in the portfolio."""

    type: Literal["table", "text"]
    data: list[dict[str, Any]]
    summary: str
    sql: str | None = None
