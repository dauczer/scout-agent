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

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/scout` | Natural language scouting query |
| `GET` | `/club-profile/{club_name}` | Club strengths/weaknesses by position |
| `GET` | `/clubs` | List all clubs in the database |
| `GET` | `/players` | Filterable, sortable player table |
| `GET` | `/health` | Health check |

### Example

```bash
curl -X POST /scout \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Nantes need to reinforce most?"}'
```

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
