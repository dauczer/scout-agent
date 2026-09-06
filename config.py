"""Centralised configuration.

Import ``get_settings()`` anywhere you need an env var.  A single
``load_dotenv()`` call happens here; every other module just imports
from this file instead of calling ``os.environ`` directly.

Missing required variables raise ``RuntimeError`` at startup so the
process fails fast with a clear message rather than crashing mid-request.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from database.constants import DEFAULT_SEASON

load_dotenv()

REQUIRED_WEB_ORIGINS = (
    "https://alexdaucourt.dev",
    "https://www.alexdaucourt.dev",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


@dataclass(frozen=True)
class Settings:
    database_url: str
    groq_api_key: str
    groq_model: str
    allowed_origins: tuple[str, ...]
    club_name: str
    club_league: str
    season: str
    scout_api_key: str | None  # optional — auth disabled when None


def get_settings() -> Settings:
    errors: list[str] = []

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        errors.append("DATABASE_URL")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        errors.append("GROQ_API_KEY")

    raw_origins = os.getenv("ALLOWED_ORIGINS", os.getenv("ALLOWED_ORIGIN", ""))
    configured_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    allowed_origins = tuple(dict.fromkeys((*REQUIRED_WEB_ORIGINS, *configured_origins)))

    if errors:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(errors)}"
        )

    return Settings(
        database_url=database_url,  # type: ignore[arg-type]  # checked above
        groq_api_key=groq_api_key,  # type: ignore[arg-type]
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        allowed_origins=allowed_origins,
        club_name=os.getenv("CLUB_NAME", "Paris S-G"),
        club_league=os.getenv("CLUB_LEAGUE", "Ligue 1"),
        season=os.getenv("SEASON", DEFAULT_SEASON),
        scout_api_key=os.getenv("SCOUT_API_KEY") or None,
    )


# Module-level singleton — evaluated once at import time.
settings = get_settings()
