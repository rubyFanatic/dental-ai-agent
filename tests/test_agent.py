"""
Agent Tests — validate core behavior before deploying.

Run: pytest tests/test_agent.py -v
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import get_agent
from app.config.practice_config import get_practice, get_all_practice_ids


# ── Config Tests ──────────────────────────────────────────────────────────

class TestPracticeConfig:
    def test_dental_practice_exists(self):
        practice = get_practice("apex-dental-01")
        assert practice["vertical"] == "dental"
        assert len(practice["services"]) > 0

    def test_medspa_practice_exists(self):
        practice = get_practice("cary-medspa-01")
        assert practice["vertical"] == "medspa"
        assert "Botox" in practice["services"]

    def test_invalid_practice_raises(self):
        with pytest.raises(KeyError):
            get_practice("nonexistent-practice")

    def test_all_practices_have_required_fields(self):
        required = ["practice_id", "vertical", "name", "services", "hours", "booking_rules", "escalation"]
        for pid in get_all_practice_ids():
            practice = get_practice(pid)
            for field in required:
                assert field in practice, f"Practice {pid} missing field: {field}"


# ── Agent Tests (require OpenAI API key) ──────────────────────────────────

@pytest.fixture(scope="module")
def agent():
    """Create agent once for all tests in this module."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    return get_agent()


class TestAgentBasic:
    def test_greeting(self, agent):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Hi")],
                "practice_id": "apex-dental-01",
                "channel": "cli",
            },
            {"configurable": {"thread_id": "test-greeting"}},
        )
        assert result.get("intent") == "greeting"
        # Should have a response
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        assert len(ai_messages) > 0

    def test_booking_intent(self, agent):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="I need a teeth cleaning next Tuesday")],
                "practice_id": "apex-dental-01",
                "channel": "cli",
            },
            {"configurable": {"thread_id": "test-booking"}},
        )
        assert result.get("intent") == "book_appointment"
        assert result.get("confidence", 0) > 0.7

    def test_escalation_request(self, agent):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="I want to talk to a real person")],
                "practice_id": "apex-dental-01",
                "channel": "cli",
            },
            {"configurable": {"thread_id": "test-escalate"}},
        )
        assert result.get("needs_escalation") == True

    def test_farewell(self, agent):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Thanks, bye!")],
                "practice_id": "apex-dental-01",
                "channel": "cli",
            },
            {"configurable": {"thread_id": "test-farewell"}},
        )
        assert result.get("intent") == "farewell"
        assert result.get("conversation_ended") == True

    def test_guardrail_no_hallucinated_service(self, agent):
        """Agent should NOT claim to offer Botox at a dental practice."""
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Can I get Botox here?")],
                "practice_id": "apex-dental-01",
                "channel": "cli",
            },
            {"configurable": {"thread_id": "test-guardrail"}},
        )
        response = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                response = msg.content.lower()
                break
        # Should NOT claim they offer Botox
        assert "book" not in response or "botox" not in response

    def test_multi_turn_context(self, agent):
        """Agent should remember context across turns."""
        thread_id = "test-multiturn"

        # Turn 1: Ask about a service
        agent.invoke(
            {
                "messages": [HumanMessage(content="Do you do teeth whitening?")],
                "practice_id": "apex-dental-01",
                "channel": "cli",
            },
            {"configurable": {"thread_id": thread_id}},
        )

        # Turn 2: Follow up without restating service
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Great, can I book that for next Friday?")],
                "practice_id": "apex-dental-01",
                "channel": "cli",
            },
            {"configurable": {"thread_id": thread_id}},
        )

        # Should recognize "that" refers to teeth whitening
        assert result.get("intent") in ["book_appointment", "ask_question"]
