import os
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

SEASON = os.getenv("SEASON", "2425")

app = FastAPI(title="Club Scout AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoutRequest(BaseModel):
    question: str


class ScoutResponse(BaseModel):
    type: str
    data: list[dict[str, Any]]
    summary: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scout", response_model=ScoutResponse)
def scout(request: ScoutRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    from agent.scout_agent import scout_query

    try:
        result = scout_query(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ScoutResponse(**result)


@app.get("/club-profile/{club_name}")
def club_profile(club_name: str):
    from database.connection import SessionLocal
    from database.models import ClubProfile

    with SessionLocal() as db:
        profiles = (
            db.query(ClubProfile)
            .filter(
                ClubProfile.club_name.ilike(f"%{club_name}%"),
                ClubProfile.season == SEASON,
            )
            .all()
        )

    if not profiles:
        raise HTTPException(
            status_code=404,
            detail=f"No club profile found for '{club_name}'",
        )

    return {
        "club_name": profiles[0].club_name,
        "league": profiles[0].league,
        "season": profiles[0].season,
        "positions": [
            {
                "position": p.position,
                "composite_score_avg": p.composite_score_avg,
                "league_composite_avg": p.league_composite_avg,
                "composite_gap": p.composite_gap,
                "league_rank": p.league_rank,
                "total_clubs": p.total_clubs,
            }
            for p in sorted(profiles, key=lambda x: x.composite_gap or 0)
        ],
    }


@app.get("/clubs")
def clubs():
    from database.connection import SessionLocal
    from database.models import Player

    with SessionLocal() as db:
        club_list = (
            db.query(Player.team, Player.league)
            .distinct()
            .order_by(Player.league, Player.team)
            .all()
        )

    return [{"name": c.team, "league": c.league} for c in club_list]


# Sortable columns for the /players endpoint
_SORTABLE = {
    "composite_score", "goals_p90", "assists_p90", "xg_p90", "xa_p90",
    "progressive_carries_p90", "progressive_passes_p90", "dribbles_completed_p90",
    "tackles_p90", "interceptions_p90", "pass_completion_pct", "shot_on_target_pct",
    "market_value_eur", "age", "minutes_played",
}


@app.get("/players")
def players(
    league: Optional[str] = None,
    position: Optional[str] = None,
    team: Optional[str] = None,
    min_minutes: int = Query(default=450, ge=0),
    sort: str = Query(default="composite_score"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Query players with optional filters. Useful for dashboard tables."""
    from database.connection import SessionLocal
    from database.models import Player

    if sort not in _SORTABLE:
        raise HTTPException(status_code=400, detail=f"Invalid sort column. Choose from: {sorted(_SORTABLE)}")

    with SessionLocal() as db:
        q = db.query(Player).filter(
            Player.season == SEASON,
            Player.minutes_played >= min_minutes,
        )
        if league:
            q = q.filter(Player.league == league)
        if position:
            q = q.filter(Player.position == position.upper())
        if team:
            q = q.filter(Player.team.ilike(f"%{team}%"))

        sort_col = getattr(Player, sort)
        q = q.order_by(sort_col.desc().nullslast())
        results = q.limit(limit).all()

    return [
        {
            "name": p.name,
            "age": p.age,
            "team": p.team,
            "league": p.league,
            "position": p.position,
            "nationality": p.nationality,
            "minutes_played": p.minutes_played,
            "matches_played": p.matches_played,
            "goals_p90": p.goals_p90,
            "assists_p90": p.assists_p90,
            "xg_p90": p.xg_p90,
            "xa_p90": p.xa_p90,
            "progressive_carries_p90": p.progressive_carries_p90,
            "progressive_passes_p90": p.progressive_passes_p90,
            "dribbles_completed_p90": p.dribbles_completed_p90,
            "tackles_p90": p.tackles_p90,
            "interceptions_p90": p.interceptions_p90,
            "pass_completion_pct": p.pass_completion_pct,
            "shot_on_target_pct": p.shot_on_target_pct,
            "composite_score": p.composite_score,
            "market_value_eur": p.market_value_eur,
            "preferred_foot": p.preferred_foot,
            "height_cm": p.height_cm,
        }
        for p in results
    ]
