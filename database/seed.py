"""Seed the database from Kaggle CSVs (FBref + Transfermarkt).

Data sources (in data/raw/):
    players_data-2024_2025.csv  — FBref stats for Big 5 leagues, 2024-25
    players.csv                 — Transfermarkt player profiles
    player_valuations.csv       — Transfermarkt historical valuations

Functions:
    seed_all               — run the full pipeline
    compute_league_averages — compute LeagueAverage rows + player z-scores + composites
    compute_club_profiles  — compute per-position ClubProfile rows for every club
"""
import logging
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from database.connection import SessionLocal, init_db
from database.constants import DEFAULT_SEASON
from database.models import ClubProfile, LeagueAverage, Player
from database.weights import ZSCORE_STATS, compute_composite

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SEASON = DEFAULT_SEASON
MIN_MINUTES = 450

# FBref Comp column → canonical league name
COMP_TO_LEAGUE: dict[str, str] = {
    "eng Premier League": "Premier League",
    "es La Liga":         "La Liga",
    "fr Ligue 1":         "Ligue 1",
    "de Bundesliga":      "Bundesliga",
    "it Serie A":         "Serie A",
}

# Transfermarkt competition_id → same canonical league names
TM_COMP_TO_LEAGUE: dict[str, str] = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "FR1": "Ligue 1",
    "L1":  "Bundesliga",
    "IT1": "Serie A",
}

# FBref CSV column name → Player model usage
# Columns at indices 0-28 are the standard section (no suffix).
# Defense/possession/passing stats have unique names (no collisions).
FBREF_COLS = {
    "name":       "Player",   # 1
    "team":       "Squad",    # 4
    "comp":       "Comp",     # 5
    "nation":     "Nation",   # 2
    "pos":        "Pos",      # 3
    "age":        "Age",      # 6
    "mp":         "MP",       # 8
    "min":        "Min",      # 10
    "nineties":   "90s",      # 11
    # Raw totals (standard section) — divide by 90s for per-90
    "goals":      "Gls",      # 12
    "assists":    "Ast",      # 13
    "xg":         "xG",       # 20
    "xag":        "xAG",      # 22
    "prgc":       "PrgC",     # 24
    "prgp":       "PrgP",     # 25
    # From specific stat sections (unique names, no collision)
    "dribbles":   "Succ",     # 150 — stats_possession
    "tackles":    "Tkl",      # 120 — stats_defense
    "interceptions": "Int",   # 131 — stats_defense
    # Direct percentages
    "cmp_pct":    "Cmp%",     # 62 — stats_passing
    "sot_pct":    "SoT%",     # 39 — stats_shooting
}

MATCH_THRESHOLD = 80

# FBref Squad name → Transfermarkt current_club_name (exact mapping)
# This eliminates fuzzy club matching — only player names need fuzzy matching.
CLUB_NAME_MAP: dict[str, str] = {
    # Premier League
    "Arsenal":          "Arsenal Football Club",
    "Aston Villa":      "Aston Villa Football Club",
    "Bournemouth":      "Association Football Club Bournemouth",
    "Brentford":        "Brentford Football Club",
    "Brighton":         "Brighton and Hove Albion Football Club",
    "Chelsea":          "Chelsea Football Club",
    "Crystal Palace":   "Crystal Palace Football Club",
    "Everton":          "Everton Football Club",
    "Fulham":           "Fulham Football Club",
    "Ipswich Town":     "Ipswich Town",
    "Leicester City":   "Leicester City",
    "Liverpool":        "Liverpool Football Club",
    "Manchester City":  "Manchester City Football Club",
    "Manchester Utd":   "Manchester United Football Club",
    "Newcastle Utd":    "Newcastle United Football Club",
    "Nott'ham Forest":  "Nottingham Forest Football Club",
    "Southampton":      "Southampton FC",
    "Tottenham":        "Tottenham Hotspur Football Club",
    "West Ham":         "West Ham United Football Club",
    "Wolves":           "Wolverhampton Wanderers Football Club",
    # La Liga
    "Alavés":           "Deportivo Alavés S. A. D.",
    "Athletic Club":    "Athletic Club Bilbao",
    "Atlético Madrid":  "Club Atlético de Madrid S.A.D.",
    "Barcelona":        "Futbol Club Barcelona",
    "Betis":            "Real Betis Balompié S.A.D.",
    "Celta Vigo":       "Real Club Celta de Vigo S. A. D.",
    "Espanyol":         "Reial Club Deportiu Espanyol de Barcelona S.A.D.",
    "Getafe":           "Getafe Club de Fútbol S. A. D. Team Dubai",
    "Girona":           "Girona Fútbol Club S. A. D.",
    "Las Palmas":       "UD Las Palmas",
    "Leganés":          "CD Leganés",
    "Mallorca":         "Real Club Deportivo Mallorca S.A.D.",
    "Osasuna":          "Club Atlético Osasuna",
    "Rayo Vallecano":   "Rayo Vallecano de Madrid S. A. D.",
    "Real Madrid":      "Real Madrid Club de Fútbol",
    "Real Sociedad":    "Real Sociedad de Fútbol S.A.D.",
    "Sevilla":          "Sevilla Fútbol Club S.A.D.",
    "Valencia":         "Valencia Club de Fútbol S. A. D.",
    "Valladolid":       "Real Valladolid CF",
    "Villarreal":       "Villarreal Club de Fútbol S.A.D.",
    # Ligue 1
    "Angers":           "Angers Sporting Club de l'Ouest",
    "Auxerre":          "Association de la Jeunesse auxerroise",
    "Brest":            "Stade brestois 29",
    "Le Havre":         "Le Havre Athletic Club",
    "Lens":             "Racing Club de Lens",
    "Lille":            "Lille Olympique Sporting Club",
    "Lyon":             "Olympique Lyonnais",
    "Marseille":        "Olympique de Marseille",
    "Monaco":           "Association sportive de Monaco Football Club",
    "Montpellier":      "Montpellier HSC",
    "Nantes":           "Football Club de Nantes",
    "Nice":             "Olympique Gymnaste Club Nice Côte d'Azur",
    "Paris S-G":        "Paris Saint-Germain Football Club",
    "Reims":            "Stade Reims",
    "Rennes":           "Stade Rennais Football Club",
    "Saint-Étienne":    "AS Saint-Étienne",
    "Strasbourg":       "Racing Club de Strasbourg Alsace",
    "Toulouse":         "Toulouse Football Club",
    # Bundesliga
    "Augsburg":         "Fußball-Club Augsburg 1907",
    "Bayern Munich":    "FC Bayern München",
    "Bochum":           "VfL Bochum",
    "Dortmund":         "Borussia Dortmund",
    "Eint Frankfurt":   "Eintracht Frankfurt Fußball AG",
    "Freiburg":         "Sport-Club Freiburg",
    "Gladbach":         "Borussia Verein für Leibesübungen 1900 Mönchengladbach",
    "Heidenheim":       "1. Fußballclub Heidenheim 1846",
    "Hoffenheim":       "Turn- und Sportgemeinschaft 1899 Hoffenheim Fußball-Spielbetriebs",
    "Holstein Kiel":    "Holstein Kiel",
    "Leverkusen":       "Bayer 04 Leverkusen Fußball",
    "Mainz 05":         "1. Fußball- und Sportverein Mainz 05",
    "RB Leipzig":       "RasenBallsport Leipzig",
    "St. Pauli":        "Fußball-Club St. Pauli von 1910",
    "Stuttgart":        "Verein für Bewegungsspiele Stuttgart 1893",
    "Union Berlin":     "1. Fußballclub Union Berlin",
    "Werder Bremen":    "Sportverein Werder Bremen von 1899",
    "Wolfsburg":        "Verein für Leibesübungen Wolfsburg",
    # Serie A
    "Atalanta":         "Atalanta Bergamasca Calcio S.p.a.",
    "Bologna":          "Bologna Football Club 1909",
    "Cagliari":         "Cagliari Calcio",
    "Como":             "Calcio Como",
    "Empoli":           "FC Empoli",
    "Fiorentina":       "Associazione Calcio Fiorentina",
    "Genoa":            "Genoa Cricket and Football Club",
    "Hellas Verona":    "Verona Hellas Football Club",
    "Inter":            "Football Club Internazionale Milano S.p.A.",
    "Juventus":         "Juventus Football Club",
    "Lazio":            "Società Sportiva Lazio S.p.A.",
    "Lecce":            "Unione Sportiva Lecce",
    "Milan":            "Associazione Calcio Milan",
    "Monza":            "AC Monza",
    "Napoli":           "Società Sportiva Calcio Napoli",
    "Parma":            "Parma Calcio 1913",
    "Roma":             "Associazione Sportiva Roma",
    "Torino":           "Torino Calcio",
    "Udinese":          "Udinese Calcio",
    "Venezia":          "Venezia FC",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if not pd.isna(f) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    f = _safe_float(val)
    return int(f) if f is not None else None


def _normalise_position(pos_raw: Optional[str]) -> str:
    if not pos_raw:
        return "MF"
    primary = str(pos_raw).split(",")[0].strip().upper()
    for code in ("FW", "MF", "DF", "GK"):
        if code in primary:
            return code
    return "MF"


def _strip_accents(s: str) -> str:
    """Remove unicode accents: Črnigoj → Crnigoj, Müller → Muller."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _fuzzy_score(fb_name: str, tm_name: str) -> float:
    """Score a player name match, with accent normalization.

    For single-word names uses fuzz.ratio (stricter) to avoid
    false positives like 'Danilo' matching 'Dani Olmo'.
    """
    a = _strip_accents(fb_name)
    b = _strip_accents(tm_name)
    if " " not in a.strip():
        return fuzz.ratio(a, b)
    return fuzz.token_sort_ratio(a, b)


# ---------------------------------------------------------------------------
# Transfermarkt loading
# ---------------------------------------------------------------------------

def _load_transfermarkt() -> pd.DataFrame:
    """Load and filter Transfermarkt CSVs, return enrichment-ready DataFrame."""
    tm_players = pd.read_csv(RAW_DIR / "players.csv", low_memory=False)

    big5_ids = set(TM_COMP_TO_LEAGUE.keys())
    tm_players = tm_players[
        (tm_players["last_season"] >= 2024)
        & (tm_players["current_club_domestic_competition_id"].isin(big5_ids))
    ].copy()

    logger.info("Transfermarkt players after filtering: %d", len(tm_players))

    # Get most recent valuation per player
    valuations = pd.read_csv(RAW_DIR / "player_valuations.csv", low_memory=False)
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    latest_vals = (
        valuations.sort_values("date", ascending=False)
        .drop_duplicates(subset=["player_id"], keep="first")[["player_id", "market_value_in_eur"]]
        .rename(columns={"market_value_in_eur": "latest_valuation_eur"})
    )

    tm_players = tm_players.merge(latest_vals, on="player_id", how="left")

    # Prefer latest valuation, fallback to profile value
    tm_players["market_value_final"] = (
        tm_players["latest_valuation_eur"].fillna(tm_players["market_value_in_eur"])
    )

    # Map competition_id to canonical league name
    tm_players["league"] = tm_players["current_club_domestic_competition_id"].map(TM_COMP_TO_LEAGUE)

    return tm_players


# ---------------------------------------------------------------------------
# Fuzzy matching FBref ↔ Transfermarkt
# ---------------------------------------------------------------------------

def _match_fbref_to_tm(
    fbref_df: pd.DataFrame,
    tm_df: pd.DataFrame,
) -> dict[tuple[str, str], dict]:
    """Return dict mapping (fbref_name, fbref_squad) → TM enrichment fields.

    Uses CLUB_NAME_MAP for exact club matching, then fuzzy-matches on player
    name only within that club's roster.
    """
    enrichment: dict[tuple[str, str], dict] = {}

    # Build TM lookup: tm_club_name → list of (player_name, enrichment_data)
    tm_by_club: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for _, row in tm_df.iterrows():
        club = str(row.get("current_club_name", "")).strip()
        name = str(row.get("name", "")).strip()
        if not name or not club:
            continue
        tm_by_club[club].append((name, {
            "market_value_eur": _safe_int(row.get("market_value_final")),
            "preferred_foot": _map_foot(row.get("foot")),
            "height_cm": _safe_int(row.get("height_in_cm")),
        }))

    # Also build a league-wide fallback for unmapped clubs
    tm_by_league: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for _, row in tm_df.iterrows():
        league = row.get("league")
        if not league:
            continue
        name = str(row.get("name", "")).strip()
        club = str(row.get("current_club_name", "")).strip()
        if not name:
            continue
        tm_by_league[league].append((name, club, {
            "market_value_eur": _safe_int(row.get("market_value_final")),
            "preferred_foot": _map_foot(row.get("foot")),
            "height_cm": _safe_int(row.get("height_in_cm")),
        }))

    matched = 0
    total = 0
    unmatched_keys: list[tuple[str, str, str]] = []  # (name, squad, league)

    # Pass 1: exact club mapping + fuzzy player name within that club
    for _, row in fbref_df.iterrows():
        comp = str(row.get(FBREF_COLS["comp"], ""))
        league = COMP_TO_LEAGUE.get(comp)
        if not league:
            continue

        fb_name = str(row.get(FBREF_COLS["name"], "")).strip()
        fb_squad = str(row.get(FBREF_COLS["team"], "")).strip()
        if not fb_name or fb_name == "nan":
            continue

        total += 1
        # Single-word names need a higher threshold to avoid false positives
        threshold = 90 if " " not in fb_name.strip() else MATCH_THRESHOLD
        tm_club = CLUB_NAME_MAP.get(fb_squad)

        if tm_club:
            candidates = tm_by_club.get(tm_club, [])
            best_score = 0.0
            best_data = None
            for tm_name, tm_data in candidates:
                score = _fuzzy_score(fb_name, tm_name)
                if score > best_score:
                    best_score = score
                    best_data = tm_data

            if best_score >= threshold and best_data is not None:
                enrichment[(fb_name, fb_squad)] = best_data
                matched += 1
            else:
                unmatched_keys.append((fb_name, fb_squad, league))
        else:
            unmatched_keys.append((fb_name, fb_squad, league))

    # Pass 2: league-wide fuzzy match for players not matched in pass 1
    # (catches transfers — player at club X in FBref but listed at club Y in TM)
    pass2_matched = 0
    for fb_name, fb_squad, league in unmatched_keys:
        threshold = 90 if " " not in fb_name.strip() else MATCH_THRESHOLD
        candidates_league = tm_by_league.get(league, [])
        best_score = 0.0
        best_data = None
        for tm_name, _tm_club, tm_data in candidates_league:
            score = _fuzzy_score(fb_name, tm_name)
            if score > best_score:
                best_score = score
                best_data = tm_data

        if best_score >= threshold and best_data is not None:
            enrichment[(fb_name, fb_squad)] = best_data
            matched += 1
            pass2_matched += 1

    logger.info(
        "Fuzzy match: %d/%d matched (%.0f%%) — %d via club, %d via league fallback",
        matched, total, 100 * matched / max(total, 1),
        matched - pass2_matched, pass2_matched,
    )
    return enrichment


def _map_foot(val) -> Optional[str]:
    if pd.isna(val) or not val:
        return None
    v = str(val).strip().lower()
    if v == "right":
        return "Right"
    if v == "left":
        return "Left"
    if v == "both":
        return "Both"
    return None


# ---------------------------------------------------------------------------
# Seed players from FBref CSV
# ---------------------------------------------------------------------------

def _seed_players(enrichment: dict[tuple[str, str], dict]) -> int:
    """Load FBref CSV, compute per-90 stats, enrich, upsert to DB. Returns count."""
    df = pd.read_csv(RAW_DIR / "players_data-2024_2025.csv", low_memory=False)
    logger.info("FBref CSV loaded: %d rows", len(df))

    rows: list[dict] = []
    for _, row in df.iterrows():
        name = str(row.get(FBREF_COLS["name"], "")).strip()
        team = str(row.get(FBREF_COLS["team"], "")).strip()
        comp = str(row.get(FBREF_COLS["comp"], ""))
        league = COMP_TO_LEAGUE.get(comp)

        if not name or name == "nan" or not team or team == "nan" or not league:
            continue

        minutes = _safe_int(row.get(FBREF_COLS["min"]))
        if minutes is None or minutes < MIN_MINUTES:
            continue

        nineties = _safe_float(row.get(FBREF_COLS["nineties"]))
        if not nineties or nineties <= 0:
            continue

        # Compute per-90 from raw totals
        def per90(col_key: str) -> Optional[float]:
            raw = _safe_float(row.get(FBREF_COLS[col_key]))
            if raw is None:
                return None
            return round(raw / nineties, 4)

        # TM enrichment
        tm = enrichment.get((name, team), {})

        rows.append({
            "name": name,
            "team": team,
            "league": league,
            "season": SEASON,
            "nationality": str(row.get(FBREF_COLS["nation"], "")).strip() or None,
            "position": _normalise_position(str(row.get(FBREF_COLS["pos"], ""))),
            "age": _safe_int(row.get(FBREF_COLS["age"])),
            "minutes_played": minutes,
            "matches_played": _safe_int(row.get(FBREF_COLS["mp"])),
            # Per-90 stats
            "goals_p90": per90("goals"),
            "assists_p90": per90("assists"),
            "xg_p90": per90("xg"),
            "xa_p90": per90("xag"),
            "progressive_carries_p90": per90("prgc"),
            "progressive_passes_p90": per90("prgp"),
            "dribbles_completed_p90": per90("dribbles"),
            "tackles_p90": per90("tackles"),
            "interceptions_p90": per90("interceptions"),
            # Direct percentages
            "pass_completion_pct": _safe_float(row.get(FBREF_COLS["cmp_pct"])),
            "shot_on_target_pct": _safe_float(row.get(FBREF_COLS["sot_pct"])),
            # TM enrichment
            "market_value_eur": tm.get("market_value_eur"),
            "preferred_foot": tm.get("preferred_foot"),
            "height_cm": tm.get("height_cm"),
        })

    if not rows:
        logger.warning("No qualifying rows found in FBref CSV")
        return 0

    with SessionLocal() as db:
        for r in rows:
            stmt = (
                sqlite_insert(Player)
                .values(**r)
                .on_conflict_do_update(
                    index_elements=["name", "team", "season"],
                    set_={k: v for k, v in r.items() if k not in ("name", "team", "season")},
                )
            )
            db.execute(stmt)
        db.commit()

    logger.info("Upserted %d players across all leagues", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# League averages + z-scores + composites (unchanged logic)
# ---------------------------------------------------------------------------

def compute_league_averages() -> None:
    """Compute LeagueAverage rows, player z-scores, and composite scores."""
    with SessionLocal() as db:
        players = (
            db.query(Player)
            .filter(Player.minutes_played >= MIN_MINUTES)
            .all()
        )

        groups: dict[tuple, list[Player]] = defaultdict(list)
        for p in players:
            groups[(p.league, p.season, p.position)].append(p)

        for (league, season, position), group in groups.items():
            avgs: dict[str, float] = {}
            stds: dict[str, float] = {}

            for field in ZSCORE_STATS:
                values = [getattr(p, field) for p in group if getattr(p, field) is not None]
                avgs[field] = float(np.mean(values)) if values else 0.0
                # Population std (ddof=0, numpy default) — deliberate choice.
                # We treat every player with ≥450 min as the full population for
                # that league/position group, not a sample drawn from a larger one.
                # Using ddof=1 (sample std) would inflate std for small groups
                # (e.g. GKs with ~30 players) and make z-scores harder to interpret.
                stds[field] = float(np.std(values)) if len(values) > 1 else 0.0

            # Compute z-scores and composite for each player
            for p in group:
                z_scores: dict[str, float] = {}
                for field in ZSCORE_STATS:
                    val = getattr(p, field)
                    std = stds[field]
                    avg = avgs[field]
                    if val is not None and std > 0 and len(group) >= 5:
                        z = round((val - avg) / std, 4)
                    else:
                        z = 0.0
                    setattr(p, f"{field}_zscore", z)
                    z_scores[field] = z
                p.composite_score = round(compute_composite(position, z_scores), 4)

            # Compute composite stats for the group (after scores are set)
            composite_vals = [p.composite_score for p in group if p.composite_score is not None]
            composite_avg = float(np.mean(composite_vals)) if composite_vals else 0.0
            composite_std = float(np.std(composite_vals)) if len(composite_vals) > 1 else 0.0

            stmt = (
                sqlite_insert(LeagueAverage)
                .values(
                    league=league,
                    season=season,
                    position=position,
                    player_count=len(group),
                    **{f"{f}_avg": avgs[f] for f in ZSCORE_STATS},
                    **{f"{f}_std": stds[f] for f in ZSCORE_STATS},
                    composite_score_avg=round(composite_avg, 4),
                    composite_score_std=round(composite_std, 4),
                )
                .on_conflict_do_update(
                    index_elements=["league", "season", "position"],
                    set_={
                        "player_count": len(group),
                        **{f"{f}_avg": avgs[f] for f in ZSCORE_STATS},
                        **{f"{f}_std": stds[f] for f in ZSCORE_STATS},
                        "composite_score_avg": round(composite_avg, 4),
                        "composite_score_std": round(composite_std, 4),
                    },
                )
            )
            db.execute(stmt)

        db.commit()

    logger.info("League averages, z-scores, and composite scores computed.")


# ---------------------------------------------------------------------------
# Club profiles (unchanged logic)
# ---------------------------------------------------------------------------

def compute_club_profiles() -> None:
    """Compute per-position ClubProfile rows for every club in the database."""
    with SessionLocal() as db:
        players = (
            db.query(Player)
            .filter(Player.minutes_played >= MIN_MINUTES)
            .all()
        )

        by_league_season: dict[tuple, dict[str, dict[str, list[Player]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for p in players:
            by_league_season[(p.league, p.season)][p.team][p.position].append(p)

        league_avgs = db.query(LeagueAverage).all()
        la_index: dict[tuple, LeagueAverage] = {
            (la.league, la.season, la.position): la for la in league_avgs
        }

        for (league, season), teams in by_league_season.items():
            positions = ["GK", "DF", "MF", "FW"]

            team_composites: dict[str, dict[str, float]] = {}
            for team, pos_map in teams.items():
                team_composites[team] = {}
                for pos in positions:
                    pos_players = pos_map.get(pos, [])
                    vals = [p.composite_score for p in pos_players if p.composite_score is not None]
                    team_composites[team][pos] = float(np.mean(vals)) if vals else 0.0

            for pos in positions:
                pos_scores = {
                    team: team_composites[team].get(pos, 0.0)
                    for team in teams
                }
                sorted_teams = sorted(pos_scores, key=pos_scores.get, reverse=True)  # type: ignore[arg-type]
                total_clubs = len(sorted_teams)

                la = la_index.get((league, season, pos))
                league_composite_avg = la.composite_score_avg if la else 0.0

                for team in teams:
                    club_avg = team_composites[team].get(pos, 0.0)
                    gap = round(club_avg - league_composite_avg, 4)
                    rank = sorted_teams.index(team) + 1

                    stmt = (
                        sqlite_insert(ClubProfile)
                        .values(
                            club_name=team,
                            league=league,
                            season=season,
                            position=pos,
                            composite_score_avg=round(club_avg, 4),
                            league_composite_avg=round(league_composite_avg, 4),
                            composite_gap=gap,
                            league_rank=rank,
                            total_clubs=total_clubs,
                        )
                        .on_conflict_do_update(
                            index_elements=["club_name", "season", "position"],
                            set_={
                                "composite_score_avg": round(club_avg, 4),
                                "league_composite_avg": round(league_composite_avg, 4),
                                "composite_gap": gap,
                                "league_rank": rank,
                                "total_clubs": total_clubs,
                            },
                        )
                    )
                    db.execute(stmt)

        db.commit()

    logger.info("Club profiles computed for all clubs.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def seed_all() -> None:
    """Full pipeline: init DB → load CSVs → enrich → seed → compute averages → club profiles."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    init_db()

    # Step 1: Load Transfermarkt data
    logger.info("Loading Transfermarkt data...")
    tm_df = _load_transfermarkt()

    # Step 2: Load FBref + fuzzy-match to TM
    logger.info("Loading FBref data and matching to Transfermarkt...")
    fbref_df = pd.read_csv(RAW_DIR / "players_data-2024_2025.csv", low_memory=False)
    enrichment = _match_fbref_to_tm(fbref_df, tm_df)

    # Step 3: Seed players
    count = _seed_players(enrichment)
    logger.info("Seeded %d players total", count)

    # Step 4: Compute league averages + z-scores + composites
    compute_league_averages()

    # Step 5: Compute club profiles
    compute_club_profiles()

    logger.info("seed_all complete.")


if __name__ == "__main__":
    seed_all()
