"""API smoke tests — exercise real routes against the committed scout.db."""


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
