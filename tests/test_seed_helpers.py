"""Unit tests for pure helper functions — no DB, no LLM."""
import os

# Ensure env vars are set before any config import (mirrors conftest.py).
os.environ.setdefault("DATABASE_URL", "sqlite:///./scout.db")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("ALLOWED_ORIGIN", "http://localhost:3000")

from database.seed import _normalise_position  # noqa: E402
from database.weights import compute_composite  # noqa: E402


class TestNormalisePosition:
    def test_fw(self):
        assert _normalise_position("FW") == "FW"

    def test_mf(self):
        assert _normalise_position("MF") == "MF"

    def test_df(self):
        assert _normalise_position("DF,MF") == "DF"

    def test_gk(self):
        assert _normalise_position("GK") == "GK"

    def test_unknown_defaults_to_mf(self):
        assert _normalise_position("AM") == "MF"

    def test_none_defaults_to_mf(self):
        assert _normalise_position(None) == "MF"


class TestComputeComposite:
    def test_zero_z_scores_gives_zero(self):
        z = {s: 0.0 for s in ["goals_p90", "xg_p90", "assists_p90", "xa_p90",
                               "dribbles_completed_p90", "progressive_carries_p90",
                               "shot_on_target_pct"]}
        assert compute_composite("FW", z) == 0.0

    def test_elite_striker_positive(self):
        # All z-scores at +2 → composite should be strongly positive
        z = {s: 2.0 for s in ["goals_p90", "xg_p90", "assists_p90", "xa_p90",
                               "dribbles_completed_p90", "progressive_carries_p90",
                               "shot_on_target_pct"]}
        score = compute_composite("FW", z)
        assert score > 1.0

    def test_weights_sum_to_one(self):
        from database.weights import POSITION_WEIGHTS
        for pos, weights in POSITION_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, f"{pos} weights sum to {total}"

    def test_unknown_position_returns_zero(self):
        assert compute_composite("XX", {"goals_p90": 2.0}) == 0.0
