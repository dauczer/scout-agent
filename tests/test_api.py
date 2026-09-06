"""API smoke tests — exercise real routes against the committed scout.db."""

import pytest


@pytest.mark.parametrize(
    "origin",
    [
        "https://alexdaucourt.dev",
        "https://www.alexdaucourt.dev",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
)
def test_preflight_allows_portfolio_and_local_origins(client, origin):
    response = client.options(
        "/scout",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,accept-language",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_preflight_rejects_unknown_origin(client):
    response = client.options(
        "/scout",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"


def test_clubs_non_empty(client):
    r = client.get("/clubs")
    assert r.status_code == 200
    clubs = r.json()
    assert len(clubs) > 0
    # Each entry has name and league keys
    assert "name" in clubs[0]
    assert "league" in clubs[0]


def test_club_profile_nantes_four_positions(client):
    r = client.get("/club-profile/Nantes")
    assert r.status_code == 200
    body = r.json()
    positions = {p["position"] for p in body["positions"]}
    assert positions == {"GK", "DF", "MF", "FW"}


def test_club_profile_ambiguous_returns_300(client):
    # "City" matches both Leicester City and Manchester City in FBref names.
    r = client.get("/club-profile/City")
    assert r.status_code == 300
    body = r.json()
    assert "matches" in body
    assert len(body["matches"]) > 1


def test_club_profile_missing_returns_404(client):
    r = client.get("/club-profile/ZZZNOCLUBZZZ")
    assert r.status_code == 404


def test_players_league_filter_limit(client):
    r = client.get("/players", params={"league": "Ligue 1", "limit": 5})
    assert r.status_code == 200
    players = r.json()
    assert len(players) == 5
    for p in players:
        assert p["league"] == "Ligue 1"


def test_scout_response_includes_sql(monkeypatch, client):
    monkeypatch.setattr(
        "api.main.scout_query",
        lambda question: {
            "type": "table",
            "data": [{"name": "Test Player", "market_value_eur": 1000000}],
            "summary": "Found one test player.",
            "sql": "SELECT name, market_value_eur FROM players LIMIT 1",
        },
    )

    response = client.post("/scout", json={"question": "find a test player"})

    assert response.status_code == 200
    body = response.json()
    assert body["sql"].startswith("SELECT")
    assert body["data"][0]["market_value_eur"] == 1000000
