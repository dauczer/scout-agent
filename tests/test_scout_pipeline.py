"""Pipeline tests with the LLM boundary replaced by deterministic plans."""
import pytest

from agent.schemas import NarrativeAnswer, QueryPlan
from agent.scout_agent import GeneratedQueryError, clear_scout_cache, scout_query


@pytest.fixture(autouse=True)
def clear_cache_between_tests():
    clear_scout_cache()
    yield
    clear_scout_cache()


def test_table_query_executes_and_returns_validated_sql(monkeypatch):
    monkeypatch.setattr(
        "agent.scout_agent._generate_plan",
        lambda question, feedback=None: QueryPlan(
            status="query",
            operation="custom_sql",
            answer_type="table",
            sql=(
                "SELECT name, team, composite_score FROM players "
                "WHERE position = 'FW' ORDER BY composite_score DESC LIMIT 3"
            ),
            club_name=None,
            message=None,
        ),
    )

    result = scout_query("top three forwards")

    assert result["type"] == "table"
    assert len(result["data"]) == 3
    assert result["sql"].endswith("LIMIT 3")
    assert isinstance(result["data"][0]["composite_score"], float)


def test_unsupported_question_does_not_execute_sql(monkeypatch):
    monkeypatch.setattr(
        "agent.scout_agent._generate_plan",
        lambda question, feedback=None: QueryPlan(
            status="unsupported",
            operation="custom_sql",
            answer_type="text",
            sql=None,
            club_name=None,
            message="Sprint speed is not available in this dataset.",
        ),
    )

    result = scout_query("fastest by sprint speed")

    assert result["type"] == "text"
    assert result["sql"] is None
    assert "not available" in result["summary"]


def test_invalid_first_plan_gets_one_retry(monkeypatch):
    plans = iter(
        [
            QueryPlan(
                status="query",
                operation="custom_sql",
                answer_type="table",
                sql="SELECT magic_score FROM secret_players",
                club_name=None,
                message=None,
            ),
            QueryPlan(
                status="query",
                operation="custom_sql",
                answer_type="table",
                sql="SELECT name FROM players ORDER BY composite_score DESC LIMIT 2",
                club_name=None,
                message=None,
            ),
        ]
    )
    feedback_seen = []

    def generate(question, feedback=None):
        feedback_seen.append(feedback)
        return next(plans)

    monkeypatch.setattr("agent.scout_agent._generate_plan", generate)

    result = scout_query("recover from invalid SQL")

    assert len(result["data"]) == 2
    assert feedback_seen[0] is None
    assert "secret_players" in feedback_seen[1]


def test_two_invalid_plans_raise(monkeypatch):
    monkeypatch.setattr(
        "agent.scout_agent._generate_plan",
        lambda question, feedback=None: QueryPlan(
            status="query",
            operation="custom_sql",
            answer_type="table",
            sql="SELECT magic_score FROM secret_players",
            club_name=None,
            message=None,
        ),
    )

    with pytest.raises(GeneratedQueryError):
        scout_query("always invalid")


def test_narrative_query_uses_optional_second_call(monkeypatch):
    monkeypatch.setattr(
        "agent.scout_agent._generate_plan",
        lambda question, feedback=None: QueryPlan(
            status="query",
            operation="custom_sql",
            answer_type="text",
            sql=(
                "SELECT position, composite_gap FROM club_profiles "
                "WHERE club_name = 'Nantes' ORDER BY composite_gap ASC LIMIT 4"
            ),
            club_name=None,
            message=None,
        ),
    )
    monkeypatch.setattr(
        "agent.scout_agent._narrate",
        lambda question, rows: NarrativeAnswer(
            text="Nantes is weakest at goalkeeper, with an important data limitation.",
            summary="Nantes' weakest measured group is goalkeeper.",
        ),
    )

    result = scout_query("What does Nantes need?")

    assert result["type"] == "text"
    assert result["sql"] is not None
    assert "Nantes" in result["data"][0]["text"]


def test_club_weakness_is_resolved_and_computed_locally(monkeypatch):
    monkeypatch.setattr(
        "agent.scout_agent._generate_plan",
        lambda question, feedback=None: QueryPlan(
            status="query",
            operation="club_weakness",
            answer_type="text",
            sql=None,
            club_name="FC Nantes",
            message=None,
        ),
    )
    monkeypatch.setattr(
        "agent.scout_agent._narrate",
        lambda question, rows: pytest.fail("club weakness must not spend a second call"),
    )

    result = scout_query("What position does FC Nantes need to reinforce most?")

    assert result["type"] == "text"
    assert result["summary"].startswith("Priority for Nantes")
    assert "Goalkeepers are excluded" in result["data"][0]["text"]
    assert "ORDER BY composite_gap ASC" in result["sql"]


def test_successful_queries_are_cached(monkeypatch):
    calls = 0

    def generate(question, feedback=None):
        nonlocal calls
        calls += 1
        return QueryPlan(
            status="query",
            operation="custom_sql",
            answer_type="table",
            sql="SELECT name FROM players ORDER BY composite_score DESC LIMIT 1",
            club_name=None,
            message=None,
        )

    monkeypatch.setattr("agent.scout_agent._generate_plan", generate)

    scout_query("same question")
    scout_query("same   question")

    assert calls == 1
