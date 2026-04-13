# Club Scout AI

Natural language scouting intelligence for Europe's Big 5 leagues.
Pick a club. Ask what it needs. Get player recommendations backed by data — not vibes.

## What it does

You ask scouting questions in plain English. The system translates them into SQL, runs them against a database of ~2,000 players, and returns structured answers with statistical context.

**Club diagnostics:**
- "What position does FC Nantes need to reinforce most?"
- "Show me Nantes' strengths and weaknesses across all positions"

**Constrained search:**
- "Find me a left-footed attacking midfielder under 25 for less than 10M"
- "I need a centre-back with strong passing, budget 15M"

**Player comparison:**
- "Compare Khvicha Kvaratskhelia to the average Serie A forward"
- "Who are the top 3 most undervalued centre-backs in Ligue 1?"

**Combined reasoning:**
- "What's Nantes' weakest position and who are the 3 best affordable options to fix it?"
- "Find a player similar to Amine Gouiri but cheaper"

## Why a SQL agent (not RAG)

This is a deliberate architectural choice, not a shortcut.

Player stats are structured data with a known schema. Every scouting question is some combination of filtering (`WHERE position = 'FW' AND age < 25`), sorting (`ORDER BY composite_score DESC`), and aggregation (`AVG`, `GROUP BY`). These are SQL's core operations.

RAG is for unstructured text where you need semantic similarity — scouting reports, match commentary, interview transcripts. Using it for tabular data would be like using a hammer as a screwdriver. A vector store can't guarantee that every result satisfies `market_value_eur < 15000000`. SQL can.

The club gap analysis — "which position group is weakest?" — requires grouping composite scores by position and ranking across clubs. Vector stores don't support `GROUP BY`.

## How the scoring works

The interesting part of this project is the reasoning layer between raw stats and scouting recommendations.

**1. Per-90 normalization.** Raw stat totals are divided by 90-minute periods played. This lets you compare a starter with 2500 minutes to a rotation player with 900.

**2. Z-score computation.** Each per-90 stat is converted to a z-score relative to the player's league and position group: `(stat - league_avg) / std_dev`. A z-score of +1.5 means "top ~7% for that stat among players in the same league and position." This makes stats immediately interpretable and comparable across leagues.

**3. Position-weighted composite scores.** Not all stats matter equally for every position. The system applies hardcoded, football-justified weights:

| Position | Key weights |
|----------|------------|
| **DF** | Tackles (0.25), interceptions (0.25), progressive passes (0.20) |
| **MF** | Progressive passes (0.20), progressive carries (0.20), xA (0.15), assists (0.15) |
| **FW** | Goals (0.25), xG (0.25), assists (0.10), xA (0.10), dribbles (0.10) |
| **GK** | Pass completion (0.50), progressive passes (0.50) — limited by available data |

Every weight is justifiable with football logic. Defenders are scored primarily on defensive actions but get credit for ball-playing ability. Forwards are scored on finishing but also on creativity and dribbling.

**4. Club gap analysis.** For each club, the system averages composite scores per position group and compares them to the league average. A negative gap means "below average at this position." The most negative gap is the priority:

```
Nantes DF gap: -0.3  (slightly below average)
Nantes MF gap: +0.2  (slightly above)
Nantes FW gap: -1.1  (well below — this is the problem)
Nantes GK gap: +0.5  (solid)
```

The agent uses this to answer "what does Nantes need?" and then searches for the best available players to fill that gap.

## Data flow

```
BUILD TIME (seed.py — runs once, offline)
==========================================

 FBref CSV ───┐
              ├── rapidfuzz ──► per-90 stats ──► z-scores ──► composite ──► SQLite
 TM CSVs ────┘   matching       + TM enrich     (pop std)    (weighted)    scout.db
              (2-pass:                                                   ┌──────────────┐
               club→league)                                              │ players       │
                                                                         │ league_avgs   │
                                                                         │ club_profiles │
                                                                         └──────────────┘

RUNTIME (per request)
======================

 Frontend ──► POST /scout ──► FastAPI ──► LangChain SQL Agent ──► SQLite (read-only)
                  │                             │
                  │                        LLM call 1:
                  │                        NL → SQL → execute → reason
                  │                             │
                  │                        LLM call 2:
                  │                        raw text → structured JSON
                  │                             │
                  ◄──── JSON { type, data, summary }
```

All expensive computation (fuzzy matching, z-scores, composites) happens at build time. The runtime agent only runs SQL queries and reasons about results.

## Tech stack

| Role | Tool | Notes |
|------|------|-------|
| Data | FBref + Transfermarkt (Kaggle CSVs) | Pre-downloaded, Big 5 leagues, 2024-25 season |
| Matching | `rapidfuzz` | Two-pass fuzzy matching: club-scoped first, league-wide fallback |
| Database | SQLite via SQLAlchemy | ~700 KB, committed to repo, zero setup |
| Agent | LangChain SQL Agent | Tool-calling architecture with read-only DB access |
| LLM | Groq (Llama 3.3 70B) | Free tier, temperature=0 for deterministic SQL generation |
| API | FastAPI | CORS, rate limiting, optional API key auth |
| Deployment | Render | SQLite ships with the repo — read-only at runtime |

## Data sources and scope

**Leagues:** Premier League, La Liga, Ligue 1, Bundesliga, Serie A

**Season:** 2024-25 (~2,000 players after 450-minute minimum filter)

**Source datasets** (in `data/raw/`):
- [Football Players Stats 2024-2025](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025) — FBref per-90 stats across 5 stat categories
- [Player Scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) — Transfermarkt profiles and historical valuations

**FBref provides:** goals, assists, xG, xA, progressive carries, progressive passes, dribbles, tackles, interceptions, pass completion %, shot on target %

**Transfermarkt enriches:** market value (EUR), preferred foot, height (cm) — matched via fuzzy matching (~85% match rate)

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/scout` | Natural language scouting query |
| `GET` | `/club-profile/{club_name}` | Club strengths/weaknesses by position |
| `GET` | `/clubs` | List all clubs in the database |
| `GET` | `/players` | Filterable, sortable player table |
| `GET` | `/health` | Health check |

### `/scout` response shape

```json
{
  "type": "table | text",
  "data": [{ "...": "..." }],
  "summary": "One-line summary of the answer"
}
```

- `"table"` — player searches, top-N lists. `data` rows share consistent keys.
- `"text"` — club diagnostics, narratives. `data` is `[{"text": "<markdown>"}]`.

### Example

```bash
curl -X POST /scout \
  -H "Content-Type: application/json" \
  -H "X-Scout-Key: your_secret_here" \
  -d '{"question": "What does Nantes need to reinforce most?"}'
```

> `X-Scout-Key` is only required when `SCOUT_API_KEY` is set in the environment.
> Leave it unset for local dev to skip auth entirely.

## Local setup

```bash
git clone <repo-url> && cd gdb_scout
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add your GROQ_API_KEY (free at console.groq.com)
uvicorn api.main:app --reload
```

That's it. `scout.db` ships with the repo — no database server, no migrations, no seed step.

### Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DATABASE_URL` | Yes | — | SQLAlchemy connection string |
| `GROQ_API_KEY` | Yes | — | LLM API key (free at console.groq.com) |
| `ALLOWED_ORIGIN` | Yes | — | Single CORS origin for the frontend |
| `SCOUT_API_KEY` | No | `None` | API key for `/scout`; auth disabled when unset |
| `CLUB_NAME` | No | `"Paris Saint-Germain"` | Default club context for the agent |
| `CLUB_LEAGUE` | No | `"Ligue 1"` | Default league context |
| `SEASON` | No | `"2425"` | Season filter |

## Reseeding the database

You don't need to do this for normal use. If you want to rebuild from scratch (e.g., new season data):

```bash
# 1. Place Kaggle CSVs in data/raw/:
#      players_data-2024_2025.csv   (FBref)
#      players.csv                   (Transfermarkt profiles)
#      player_valuations.csv         (Transfermarkt valuations)

# 2. Run the seed pipeline (idempotent — safe to re-run):
python -c "from database.seed import seed_all; seed_all()"

# 3. Commit the updated scout.db
git add scout.db && git commit -m "reseed: 2024-25 data"
```

## Security model

The agent connects to SQLite via `?mode=ro&immutable=1` — a driver-level guarantee that DML is rejected before execution. This isn't a prompt asking the LLM to behave; it's a hard technical enforcement. Even if someone prompt-injects the agent into generating `DROP TABLE`, the sqlite3 driver returns an error.

Other layers: CORS locked to a single origin, rate limiting (10/min on `/scout`), optional API key auth, input validation (3-500 chars), 50-second request timeout, generic error messages (no stack traces leaked).

## Known limitations

**GK composite scores are incomplete.** FBref outfield stats don't include saves, clean sheets, or shot-stopping metrics. The GK composite only measures distribution quality via passing. The agent prompt acknowledges this and de-prioritizes GK findings.

**~15% of players lack Transfermarkt data.** Fuzzy matching between FBref and Transfermarkt covers ~85% of players. The rest still have all FBref stats but no market value, preferred foot, or height. Budget-filtered queries exclude these players by necessity.

**Four broad position groups.** The data layer uses GK, DF, MF, FW. It doesn't distinguish centre-backs from fullbacks or attacking midfielders from defensive midfielders. The agent compensates in its reasoning — when you ask for a "ball-playing CB," it knows to emphasize progressive passes within the DF group — but the composite scores can't capture sub-role nuance.

**The 450-minute filter removes young prospects.** Promising players with limited minutes are excluded to reduce statistical noise. This is a trade-off: reliability over discovery.

## In production, this would be...

This is a portfolio project. Here's what would change at scale:

| This project | Production equivalent |
|-------------|---------------------|
| Static Kaggle CSVs | Airflow DAG pulling from live APIs on a schedule |
| `rapidfuzz` matching | Probabilistic record linkage (Splink) with human-in-the-loop review |
| Hardcoded position weights | Feature store (Feast/Tecton) with versioning and A/B testing |
| SQLite committed to repo | PostgreSQL with read replicas (same SQLAlchemy code, just swap `DATABASE_URL`) |
| Groq free tier | Azure OpenAI / Bedrock with fallback routing and cost monitoring |
| `verbose=True` stdout logs | LLM observability (LangSmith, Helicone) with trace IDs and eval suites |
| Optional shared API key | OAuth2/JWT with per-user identity and audit logging |
| In-process rate limiting | Distributed rate limiter (Redis-backed or API Gateway-level) |
| No query cache | Redis cache keyed by question hash — repeated questions skip LLM calls entirely |
