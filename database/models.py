from sqlalchemy import Column, Integer, String, Float, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    team = Column(String, nullable=False)
    league = Column(String, nullable=False)   # "FRA-Ligue 1", "ENG-Premier League", etc.
    season = Column(String, nullable=False)   # "2324", "2425"
    nationality = Column(String)
    position = Column(String)                 # "FW", "MF", "DF", "GK"
    age = Column(Integer)

    # Transfermarkt enrichment (nullable)
    preferred_foot = Column(String)           # "Left", "Right", "Both"
    height_cm = Column(Integer)
    market_value_eur = Column(Integer)

    # Volume — critical for reliability
    minutes_played = Column(Integer)
    matches_played = Column(Integer)

    # Per-90 stats
    goals_p90 = Column(Float)
    assists_p90 = Column(Float)
    xg_p90 = Column(Float)
    xa_p90 = Column(Float)
    progressive_carries_p90 = Column(Float)
    progressive_passes_p90 = Column(Float)
    dribbles_completed_p90 = Column(Float)
    tackles_p90 = Column(Float)
    interceptions_p90 = Column(Float)
    pass_completion_pct = Column(Float)       # percentage, not per-90
    shot_on_target_pct = Column(Float)        # percentage, not per-90

    # Z-scores vs league average for the player's position
    goals_p90_zscore = Column(Float)
    assists_p90_zscore = Column(Float)
    xg_p90_zscore = Column(Float)
    xa_p90_zscore = Column(Float)
    progressive_carries_p90_zscore = Column(Float)
    progressive_passes_p90_zscore = Column(Float)
    dribbles_completed_p90_zscore = Column(Float)
    tackles_p90_zscore = Column(Float)
    interceptions_p90_zscore = Column(Float)
    pass_completion_pct_zscore = Column(Float)
    shot_on_target_pct_zscore = Column(Float)

    # Composite score — single position-weighted number
    composite_score = Column(Float)

    __table_args__ = (
        UniqueConstraint("name", "team", "season", name="uq_player_team_season"),
    )


class LeagueAverage(Base):
    __tablename__ = "league_averages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    league = Column(String, nullable=False)
    season = Column(String, nullable=False)
    position = Column(String, nullable=False)  # "FW", "MF", "DF", "GK"
    player_count = Column(Integer)

    goals_p90_avg = Column(Float)
    goals_p90_std = Column(Float)
    assists_p90_avg = Column(Float)
    assists_p90_std = Column(Float)
    xg_p90_avg = Column(Float)
    xg_p90_std = Column(Float)
    xa_p90_avg = Column(Float)
    xa_p90_std = Column(Float)
    progressive_carries_p90_avg = Column(Float)
    progressive_carries_p90_std = Column(Float)
    progressive_passes_p90_avg = Column(Float)
    progressive_passes_p90_std = Column(Float)
    dribbles_completed_p90_avg = Column(Float)
    dribbles_completed_p90_std = Column(Float)
    tackles_p90_avg = Column(Float)
    tackles_p90_std = Column(Float)
    interceptions_p90_avg = Column(Float)
    interceptions_p90_std = Column(Float)
    pass_completion_pct_avg = Column(Float)
    pass_completion_pct_std = Column(Float)
    shot_on_target_pct_avg = Column(Float)
    shot_on_target_pct_std = Column(Float)

    # Average composite score for this position group
    composite_score_avg = Column(Float)
    composite_score_std = Column(Float)

    __table_args__ = (
        UniqueConstraint("league", "season", "position", name="uq_leagueavg"),
    )


class ClubProfile(Base):
    """One row per (club, season, position) — 4 rows per club per season."""
    __tablename__ = "club_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    club_name = Column(String, nullable=False)
    league = Column(String, nullable=False)
    season = Column(String, nullable=False)
    position = Column(String, nullable=False)  # "GK", "DF", "MF", "FW"

    # Average composite score of this club's players at this position
    composite_score_avg = Column(Float)

    # League average composite for this position (copied for easy comparison)
    league_composite_avg = Column(Float)

    # Gap: club - league average (negative = below average = needs improvement)
    composite_gap = Column(Float)

    # Rank among all clubs in the league for this position (1 = best)
    league_rank = Column(Integer)
    total_clubs = Column(Integer)

    __table_args__ = (
        UniqueConstraint("club_name", "season", "position", name="uq_club_season_position"),
    )
