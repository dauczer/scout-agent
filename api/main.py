import asyncio
import hashlib
import logging
import time
from functools import lru_cache
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text

from agent.scout_agent import scout_query
from config import settings
from database.connection import SessionLocal, engine
from database.models import ClubProfile, Player

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("scout.api")

SEASON = settings.season

_api_key_header = APIKeyHeader(name="X-Scout-Key", auto_error=False)


def _require_scout_key(key: str | None = Depends(_api_key_header)) -> None:
    """Dependency that enforces X-Scout-Key on protected routes."""
    if not settings.scout_api_key:
        # Key not configured — auth is disabled (useful for local dev).
        return
    if key != settings.scout_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Scout-Key")


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Club Scout AI")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origin],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ScoutRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class ScoutResponse(BaseModel):
    type: str
    data: list[dict[str, Any]]
    summary: str


@app.get("/health")
@limiter.limit("60/minute")
def health(request: Request):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        logger.exception("health check DB probe failed")
        db_status = "error"

    status_code = 200 if db_status == "ok" else 503
    return JSONResponse(
        content={"status": "ok" if db_status == "ok" else "degraded", "db": db_status},
        status_code=status_code,
    )


@app.post("/scout", response_model=ScoutResponse)
@limiter.limit("10/minute")
async def scout(request: Request, body: ScoutRequest, _: None = Depends(_require_scout_key)):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    q_hash = hashlib.sha256(body.question.encode()).hexdigest()[:8]
    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(scout_query, body.question),
            timeout=50,
        )
        logger.info("scout ok q=%s latency=%.2fs", q_hash, time.monotonic() - t0)
    except asyncio.TimeoutError:
        logger.warning("scout timeout q=%s latency=%.2fs", q_hash, time.monotonic() - t0)
        raise HTTPException(status_code=504, detail="The scouting query timed out. Please try a simpler question.")
    except Exception:
        logger.exception("scout error q=%s latency=%.2fs", q_hash, time.monotonic() - t0)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

    return ScoutResponse(**result)


@app.get("/club-profile/{club_name}")
@limiter.limit("60/minute")
def club_profile(request: Request, club_name: str):
    with SessionLocal() as db:
        # Exact case-insensitive match first.
        exact = (
            db.query(ClubProfile)
            .filter(
                ClubProfile.club_name.ilike(club_name),
                ClubProfile.season == SEASON,
            )
            .all()
        )
        if exact:
            profiles = exact
        else:
            # Fall back to substring search for disambiguation.
            candidates = (
                db.query(ClubProfile.club_name)
                .filter(
                    ClubProfile.club_name.ilike(f"%{club_name}%"),
                    ClubProfile.season == SEASON,
                )
                .distinct()
                .all()
            )
            distinct_names = [r.club_name for r in candidates]

            if not distinct_names:
                raise HTTPException(
                    status_code=404,
                    detail=f"No club profile found for '{club_name}'",
                )
            if len(distinct_names) > 1:
                return JSONResponse(
                    status_code=300,
                    content={
                        "detail": f"'{club_name}' matched multiple clubs. Use an exact name.",
                        "matches": distinct_names,
                    },
                )
            # Exactly one fuzzy match — load its rows.
            profiles = (
                db.query(ClubProfile)
                .filter(
                    ClubProfile.club_name == distinct_names[0],
                    ClubProfile.season == SEASON,
                )
                .all()
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


@lru_cache(maxsize=1)
def _clubs_cached() -> list[dict]:
    """Clubs list never changes between deploys — cache it after the first DB hit."""
    with SessionLocal() as db:
        club_list = (
            db.query(Player.team, Player.league)
            .distinct()
            .order_by(Player.league, Player.team)
            .all()
        )
    return [{"name": c.team, "league": c.league} for c in club_list]


@app.get("/clubs")
@limiter.limit("60/minute")
def clubs(request: Request):
    return _clubs_cached()


# Sortable columns for the /players endpoint
_SORTABLE = {
    "composite_score", "goals_p90", "assists_p90", "xg_p90", "xa_p90",
    "progressive_carries_p90", "progressive_passes_p90", "dribbles_completed_p90",
    "tackles_p90", "interceptions_p90", "pass_completion_pct", "shot_on_target_pct",
    "market_value_eur", "age", "minutes_played",
}


@app.get("/players")
@limiter.limit("60/minute")
def players(
    request: Request,
    league: Optional[str] = None,
    position: Optional[str] = None,
    team: Optional[str] = None,
    min_minutes: int = Query(default=450, ge=0),
    sort: str = Query(default="composite_score"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Query players with optional filters. Useful for dashboard tables."""
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
