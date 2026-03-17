"""
LangGraph Agent Construction.

This is where the graph gets assembled:
- Nodes are added (each is a function from nodes.py)
- Edges define the flow between nodes
- Conditional edges handle routing logic

The compiled graph is the "brain" of the agent.
It handles multi-turn conversations with state persistence.

Architecture:
  Intent Classification → Routing → Tool Execution → Response → Guardrail → Output
"""
import logging
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agent.state import AgentState
from app.agent.nodes import (
    classify_intent,
    respond_with_tools,
    execute_tools,
    check_guardrails,
    handle_escalation,
    handle_greeting,
    handle_farewell,
)

logger = logging.getLogger(__name__)


def build_agent(checkpointer=None):
    """
    Build and compile the LangGraph agent.

    Args:
        checkpointer: State persistence backend.
                      - MemorySaver() for dev/testing
                      - PostgresSaver() for production

    Returns:
        Compiled LangGraph application ready to invoke.
    """

    # ── Create the graph ──────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_farewell", handle_farewell)
    graph.add_node("handle_escalation", handle_escalation)
    graph.add_node("respond_with_tools", respond_with_tools)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("check_guardrails", check_guardrails)

    # ── Set entry point ───────────────────────────────────────────────
    graph.set_entry_point("classify_intent")

    # ── Conditional edge: Route by intent ─────────────────────────────
    # After intent classification, decide where to go next
    graph.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {
            "greeting": "handle_greeting",
            "farewell": "handle_farewell",
            "escalation": "handle_escalation",
            "conversation": "respond_with_tools",
        },
    )

    # ── Conditional edge: After LLM response ──────────────────────────
    # The LLM might want to call a tool or respond directly
    graph.add_conditional_edges(
        "respond_with_tools",
        _route_after_llm,
        {
            "tool_call": "execute_tools",
            "guardrail_check": "check_guardrails",
        },
    )

    # ── After tool execution, go back to LLM for response ─────────────
    # The LLM needs to see the tool results to formulate a response
    graph.add_edge("execute_tools", "respond_with_tools")

    # ── After guardrail check ─────────────────────────────────────────
    graph.add_conditional_edges(
        "check_guardrails",
        _route_after_guardrail,
        {
            "pass": END,
            "violation": "respond_with_tools",  # Regenerate response
        },
    )

    # ── Terminal nodes go to END ──────────────────────────────────────
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_farewell", END)
    graph.add_edge("handle_escalation", END)

    # ── Compile with checkpointer ─────────────────────────────────────
    if checkpointer is None:
        checkpointer = MemorySaver()

    app = graph.compile(checkpointer=checkpointer)

    logger.info("Agent graph compiled successfully")
    return app


# ── Routing Functions ─────────────────────────────────────────────────────

def _route_by_intent(state: AgentState) -> str:
    """
    Route based on classified intent.

    This is the main routing decision in the graph.
    Maps intents to the next node.
    """
    intent = state.get("intent", "unclear")
    needs_escalation = state.get("needs_escalation", False)

    # Escalation takes priority
    if needs_escalation:
        logger.info(f"Routing to escalation. Reason: {state.get('escalation_reason', 'unknown')}")
        return "escalation"

    # Route by intent
    routing = {
        "greeting": "greeting",
        "farewell": "farewell",
        "escalate_to_human": "escalation",
        "book_appointment": "conversation",
        "reschedule": "conversation",
        "cancel": "conversation",
        "ask_question": "conversation",
        "insurance_question": "conversation",
        "unclear": "conversation",  # LLM will ask a clarifying question
    }

    route = routing.get(intent, "conversation")
    logger.info(f"Intent: {intent} (confidence: {state.get('confidence', 0):.2f}) → Routing to: {route}")
    return route


def _route_after_llm(state: AgentState) -> str:
    """
    After the LLM responds, check if it wants to call tools.

    If the response contains tool_calls → execute them.
    If it's a direct text response → check guardrails before sending.
    """
    messages = state.get("messages", [])
    if not messages:
        return "guardrail_check"

    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_call"

    return "guardrail_check"


def _route_after_guardrail(state: AgentState) -> str:
    """
    After guardrail check, either pass through or regenerate.

    If violation detected → send back to LLM to try again.
    If clean → send to patient (END).
    """
    if state.get("guardrail_violation", False):
        logger.warning("Guardrail violation detected — regenerating response")
        return "violation"

    return "pass"


# ── Convenience: Get a ready-to-use agent ─────────────────────────────────

def get_agent():
    """Get a compiled agent with in-memory checkpointer. Good for dev/testing."""
    return build_agent(checkpointer=MemorySaver())
