"""Response schemas for the scouting agent."""
from typing import Any

from pydantic import BaseModel


class ScoutResponse(BaseModel):
    """Structured response from the scouting agent.

    type values:
        "player_list"        — ranked list of players (e.g. recommendations, top N)
        "club_profile"       — club strengths/weaknesses by position
        "player_comparison"  — side-by-side comparison of 2+ players
        "general"            — free-form answer that doesn't fit the above
    """
    type: str
    data: list[dict[str, Any]]
    summary: str
