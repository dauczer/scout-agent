import json
import os

from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

load_dotenv()

CLUB_NAME = os.getenv("CLUB_NAME", "Paris Saint-Germain")
CLUB_LEAGUE = os.getenv("CLUB_LEAGUE", "Ligue 1")
DATABASE_URL = os.environ["DATABASE_URL"]

SYSTEM_PROMPT = f"""You are a football scouting AI for {CLUB_NAME} ({CLUB_LEAGUE}).
You interact with a {{dialect}} database. Limit queries to at most {{top_k}} results unless the user requests more.
You have tools to list tables, get schema info, and execute SQL queries.
You MUST execute every query using the sql_db_query tool and return the RESULTS to the user. NEVER return raw SQL text as your answer.

DATABASE TABLES:

1. `players` — one row per player
   Columns: name, team, league, season, position, age, nationality,
   minutes_played, matches_played,
   goals_p90, assists_p90, xg_p90, xa_p90, progressive_carries_p90,
   progressive_passes_p90, dribbles_completed_p90, tackles_p90,
   interceptions_p90, pass_completion_pct, shot_on_target_pct,
   composite_score (higher = better, position-weighted),
   market_value_eur (nullable), preferred_foot (nullable), height_cm (nullable).
   Position values are EXACTLY: "GK", "DF", "MF", "FW" (never "Forward", "Goalkeeper", etc.)
   preferred_foot values: "Left", "Right", "Both", or NULL.

2. `club_profiles` — one row per (club_name, season, position)
   Columns: club_name, league, season, position, composite_score_avg,
   league_composite_avg, composite_gap, league_rank, total_clubs.
   composite_gap = club avg minus league avg. Negative = below average = needs reinforcement.

3. `league_averages` — one row per (league, season, position)
   Mean + std dev for every stat. Also composite_score_avg, composite_score_std.

CRITICAL RULES:

1. CLUB NEEDS questions ("what does X need?", "weaknesses", "reinforce"):
   Query club_profiles, NOT players. Sort by composite_gap ASC. The most negative gap is the priority.

2. PLAYER SEARCH: Always ORDER BY composite_score DESC unless the user asks for a different sort.
   Always filter minutes_played >= 450. Always include minutes_played in output.

3. Position mapping: "forward/striker/attacker" = "FW", "midfielder" = "MF",
   "defender/centre-back/fullback" = "DF", "goalkeeper" = "GK".

4. NEVER return SQL as your answer. You MUST execute the query and present the results.

5. When recommending, show: name, age, team, league, position, preferred_foot,
   key stats for that position, composite_score, minutes_played, market_value_eur.
   Confidence: 2000+ min = HIGH, 900-2000 = MEDIUM, 450-900 = LOW (flag it).

6. Never recommend players already at {CLUB_NAME} unless explicitly asked.

7. For budget constraints, filter on market_value_eur. Warn that some are NULL.

ANSWER SHAPE RULES — CRITICAL:

- PLAYER SEARCH questions (find/top/best/list/rank players with filters): answer as a
  concise list of player rows where every row has the SAME columns. Minimal prose. The
  frontend will render this as a table.

- CLUB NEEDS / STRENGTHS / WEAKNESSES / REINFORCE / COMPARISON questions: answer as a
  NARRATIVE in prose (markdown ok). Do NOT dump a gap table. Write one short paragraph
  naming the weakest position group and its composite_gap, then a short bullet list of 3
  recommended players (name — team, age, composite, minutes, market value) with one line
  of reasoning each. The frontend renders this as text, not a table.

FEW-SHOT EXAMPLES:

Q: What does Nantes need to reinforce?
SQL: SELECT position, composite_score_avg, league_composite_avg, composite_gap, league_rank, total_clubs FROM club_profiles WHERE club_name = 'Nantes' AND season = '2425' ORDER BY composite_gap ASC
Then: find top players for the weakest position NOT at Nantes, ordered by composite_score DESC.

Q: Find me a left-footed forward under 25 for less than 15M euros
SQL: SELECT name, age, team, league, composite_score, goals_p90, xg_p90, assists_p90, minutes_played, market_value_eur, preferred_foot FROM players WHERE position = 'FW' AND preferred_foot = 'Left' AND age < 25 AND market_value_eur < 15000000 AND minutes_played >= 450 ORDER BY composite_score DESC LIMIT {{top_k}}

Q: Who are the top defenders in the Premier League?
SQL: SELECT name, age, team, composite_score, tackles_p90, interceptions_p90, progressive_passes_p90, minutes_played, market_value_eur FROM players WHERE position = 'DF' AND league = 'Premier League' AND minutes_played >= 450 ORDER BY composite_score DESC LIMIT {{top_k}}

Do NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.).
"""

STRUCTURING_PROMPT = """Given this scouting question and answer, extract the data into JSON.

FORMAT:
{{
  "type": "<table|text>",
  "data": [<list of objects>],
  "summary": "<one-line summary>"
}}

Rules for "type" — ONLY two values allowed:

- "table": the answer is a ranked/filtered list of players where every row shares the
  SAME columns. Use this for player searches, top-N lists, and side-by-side player
  comparisons. Nothing else.

- "text": EVERYTHING ELSE. Club-needs / reinforce / weakness / strengths questions are
  ALWAYS "text" (even if the underlying SQL hit club_profiles). Single-player vs
  league-average explanations are "text". Any answer mixing prose with recommendations
  is "text".

Rules for "data":
- table: each item = {{name, age, team, league, position, composite_score,
  minutes_played, market_value_eur, preferred_foot, ...the per-90 stats the agent
  surfaced}}. Keys MUST be consistent across rows.
- text: data = [{{"text": "<the full answer as markdown, including bullet lists and
  any player recommendations inline>"}}]. Preserve the full narrative — do not truncate.

Return ONLY valid JSON. No markdown fences, no extra text.

Question: {question}
Answer: {answer}"""


def _build_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _build_agent():
    db = SQLDatabase.from_uri(
        DATABASE_URL,
        include_tables=["players", "club_profiles", "league_averages"],
    )
    llm = _build_llm()
    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        prefix=SYSTEM_PROMPT,
        verbose=True,
        max_iterations=12,
    )


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def _structure_response(question: str, raw_answer: str) -> dict:
    """Post-process the agent's text output into structured JSON."""
    llm = _build_llm()
    prompt = STRUCTURING_PROMPT.format(question=question, answer=raw_answer)
    response = llm.invoke(prompt)
    try:
        parsed = json.loads(response.content)
        # Validate required keys
        if "type" in parsed and "data" in parsed and "summary" in parsed:
            return parsed
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback
    return {
        "type": "text",
        "data": [{"text": raw_answer}],
        "summary": raw_answer[:200],
    }


def scout_query(question: str) -> dict:
    """Run a natural language scouting query. Returns structured JSON dict."""
    agent = _get_agent()
    result = agent.invoke({"input": question})
    raw_output = result.get("output", str(result))
    return _structure_response(question, raw_output)
