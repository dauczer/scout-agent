"""Test configuration.

Env vars must be set before any app module is imported, because config.py
validates them at import time (module-level singleton).
"""
import os

import pytest

# Set required env vars before any import of config / api / agent.
os.environ.setdefault("DATABASE_URL", "sqlite:///./scout.db")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used-in-unit-tests")
os.environ.setdefault("ALLOWED_ORIGIN", "http://localhost:3000")
# Leave SCOUT_API_KEY unset so auth is disabled in tests.


@pytest.fixture(scope="session")
def client():
    """Return a TestClient for the FastAPI app.

    Imported here (not at module level) so the env vars above are in place
    before api.main is loaded.
    """
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)
