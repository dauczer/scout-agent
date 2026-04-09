# Club Scout AI

Natural language scouting intelligence for Europe's Big 5 leagues.
Pick a club. Ask what it needs. Get player recommendations backed by data.

## How it works

Pre-downloaded FBref and Transfermarkt datasets are fuzzy-matched, normalized into z-scores, and stored in SQLite. A LangChain SQL agent translates natural language scouting questions into precise database queries, using position-weighted composite scores to identify club weaknesses and recommend transfer targets.

**This is a SQL agent, not RAG.** Player stats are structured data -- SQL is the right tool for the job.

## Tech stack

| Role | Tool |
|------|------|
| Data | FBref stats + Transfermarkt valuations (Kaggle) |
| Matching | `rapidfuzz` fuzzy name matching |
| Database | SQLite via SQLAlchemy |
| Agent | LangChain SQL Agent |
| LLM | Groq (Llama 3.3 70B) |
| API | FastAPI |
| Deployment | Render |

## Data sources

Raw CSV data (not included in repo) from Kaggle:
- [Football Players Stats 2024-2025](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025) -- FBref Big 5 league stats
- [Player Scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) -- Transfermarkt profiles and valuations

### Reseeding the database

`scout.db` is pre-built and ships with the repo — you don't need to reseed for normal use.

If you want to rebuild it from scratch (e.g. to update to a new season):

```bash
# 1. Download the two Kaggle datasets above and place them in data/raw/:
#      data/raw/players_data-2024_2025.csv   (FBref stats)
#      data/raw/players.csv                   (Transfermarkt profiles)
#      data/raw/player_valuations.csv          (Transfermarkt valuations)

# 2. Run the seed pipeline (idempotent — safe to re-run):
python -c "from database.seed import seed_all; seed_all()"

# 3. Commit the updated scout.db
git add scout.db && git commit -m "reseed: 2024-25 data"
```

> After reseeding, DB indexes defined in `database/models.py` will be applied automatically.
> They are not retroactively added to an existing `scout.db`.

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

- `"table"` — player searches / top-N lists. `data` rows share consistent keys.
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
# Clone and enter
git clone <repo-url> && cd club-scout-ai

# Virtual environment
python -m venv .venv && source .venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env: add your GROQ_API_KEY (free at console.groq.com)

# Run (scout.db is pre-seeded and ships with the repo)
uvicorn api.main:app --reload
```

## Leagues covered

Premier League, La Liga, Ligue 1, Bundesliga, Serie A -- 2024-25 season.
