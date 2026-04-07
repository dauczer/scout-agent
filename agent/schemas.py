"""Response schemas for the scouting agent."""
from typing import Any

from pydantic import BaseModel


class ScoutResponse(BaseModel):
    """Structured response from the scouting agent.

    type values (only two):
        "table" — `data` is a list of homogeneous player row dicts with consistent
                  keys (player searches, top-N, side-by-side comparisons).
        "text"  — `data` is `[{"text": "<markdown>"}]`. Used for club-needs diagnostics,
                  strengths/weaknesses, narrative answers, anything non-tabular.
    """
    type: str
    data: list[dict[str, Any]]
    summary: str
