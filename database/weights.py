"""Position-specific stat weights for computing composite player scores.

Every weight is justified by football logic — see CLAUDE.md for the rationale.
These weights are the single source of truth used by seed.py and the agent prompt.
"""

POSITION_WEIGHTS: dict[str, dict[str, float]] = {
    "GK": {
        # FBref outfield stats don't capture GK-specific metrics (saves, clean sheets).
        # We use passing as a proxy for distribution quality — known limitation.
        "pass_completion_pct":      0.50,
        "progressive_passes_p90":   0.50,
    },
    "DF": {
        "tackles_p90":              0.25,
        "interceptions_p90":        0.25,
        "progressive_passes_p90":   0.20,  # ball-playing ability
        "progressive_carries_p90":  0.10,
        "pass_completion_pct":      0.10,
        "goals_p90":                0.05,
        "assists_p90":              0.05,
    },
    "MF": {
        "progressive_passes_p90":   0.20,
        "progressive_carries_p90":  0.20,
        "xa_p90":                   0.15,
        "assists_p90":              0.15,
        "pass_completion_pct":      0.10,
        "tackles_p90":              0.10,
        "goals_p90":                0.05,
        "xg_p90":                   0.05,
    },
    "FW": {
        "goals_p90":                0.25,
        "xg_p90":                   0.25,
        "assists_p90":              0.10,
        "xa_p90":                   0.10,
        "dribbles_completed_p90":   0.10,
        "progressive_carries_p90":  0.10,
        "shot_on_target_pct":       0.10,
    },
}

# Stat columns that have z-score counterparts in the Player model
ZSCORE_STATS: list[str] = [
    "goals_p90",
    "assists_p90",
    "xg_p90",
    "xa_p90",
    "progressive_carries_p90",
    "progressive_passes_p90",
    "dribbles_completed_p90",
    "tackles_p90",
    "interceptions_p90",
    "pass_completion_pct",
    "shot_on_target_pct",
]


def compute_composite(position: str, z_scores: dict[str, float]) -> float:
    """Return the weighted composite score for a player given their position and z-scores."""
    weights = POSITION_WEIGHTS.get(position, {})
    return sum(weights.get(stat, 0.0) * z_scores.get(stat, 0.0) for stat in weights)
