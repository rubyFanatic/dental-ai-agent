"""
LangGraph Node Implementations.

Each node is a function that:
1. Receives the current AgentState
2. Does its work (LLM call, tool execution, validation)
3. Returns a dict of state fields to update

LangGraph automatically merges the returned fields into the state.
"""
import json
import logging
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from app.agent.prompts import build_system_prompt, INTENT_CLASSIFICATION_PROMPT
from app.agent.state import AgentState
from app.config.practice_config import get_practice
from app.config.settings import get_settings
from app.tools.definitions import TOOLS
from app.tools.calendar import execute_tool

logger = logging.getLogger(__name__)

# ── LLM Setup ─────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Get configured LLM instance."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


# ── Node 1: Intent Classifier ─────────────────────────────────────────────

def classify_intent(state: AgentState) -> dict:
    """
    Classify the patient's intent from their latest message.

    Returns: intent, confidence, extracted entities.
    Uses a dedicated LLM call with structured output.
    """
    practice = get_practice(state["practice_id"])
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""

    # Build conversation history for context
    history = "\n".join(
        f"{'Patient' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in messages[-6:]  # Last 3 turns of context
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
    )

    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        services=", ".join(practice["services"]),
        message=latest_message,
        history=history,
    )

    llm = _get_llm(temperature=0.1)  # Low temp for classification
    response = llm.invoke([
        SystemMessage(content="You are an intent classifier. Respond ONLY with valid JSON. No markdown, no backticks."),
        HumanMessage(content=prompt),
    ])

    # Parse the classification
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse intent classification: {response.content}")
        result = {
            "intent": "unclear",
            "confidence": 0.3,
            "extracted_service": None,
            "extracted_date": None,
            "extracted_name": None,
        }

    # Check confidence against escalation threshold
    escalation_threshold = practice["escalation"]["confidence_threshold"]
    needs_escalation = result.get("confidence", 0) < escalation_threshold

    return {
        "intent": result.get("intent", "unclear"),
        "confidence": result.get("confidence", 0.0),
        "extracted_service": result.get("extracted_service") or "",
        "extracted_date": result.get("extracted_date") or "",
        "extracted_name": result.get("extracted_name") or "",
        "needs_escalation": needs_escalation or result.get("intent") == "escalate_to_human",
        "escalation_reason": "Low confidence" if needs_escalation else "",
        "turn_count": state.get("turn_count", 0) + 1,
        "last_node": "classify_intent",
    }


# ── Node 2: Respond with Tools (Main Conversation Node) ───────────────────

def respond_with_tools(state: AgentState) -> dict:
    """
    Generate a response using the LLM with function calling tools.

    The LLM decides whether to:
    - Respond directly (for greetings, simple questions)
    - Call a tool (for bookings, availability checks, service lookups)
    - Ask a clarifying question

    This is the main conversation node — it handles the bulk of interactions.
    """
    practice = get_practice(state["practice_id"])
    system_prompt = build_system_prompt(practice)

    # Replace {channel} placeholder in system prompt
    system_prompt = system_prompt.replace("{channel}", state.get("channel", "text"))

    llm = _get_llm(temperature=0.3)
    llm_with_tools = llm.bind_tools(TOOLS)

    # Build message list for the LLM
    llm_messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    # Call LLM with tools available
    response = llm_with_tools.invoke(llm_messages)

    return {
        "messages": [response],
        "last_node": "respond_with_tools",
    }


# ── Node 3: Execute Tool Calls ─────────────────────────────────────────────

def execute_tools(state: AgentState) -> dict:
    """
    Execute any tool calls the LLM made and return results.

    When the LLM decides to call a function (e.g., check_availability),
    this node executes the actual function and feeds the result back.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"last_node": "execute_tools"}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Inject practice_id if the tool needs it but LLM didn't provide it
        if "practice_id" in str(TOOLS) and "practice_id" not in tool_args:
            tool_args["practice_id"] = state["practice_id"]

        logger.info(f"Executing tool: {tool_name}({tool_args})")

        result = execute_tool(tool_name, tool_args)

        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

        # Track booking confirmations
        if tool_name == "book_appointment":
            result_data = json.loads(result)
            if result_data.get("success"):
                return {
                    "messages": tool_messages,
                    "booking_confirmed": True,
                    "booking_id": result_data.get("booking_id", ""),
                    "last_node": "execute_tools",
                }

    return {
        "messages": tool_messages,
        "last_node": "execute_tools",
    }


# ── Node 4: Guardrail Check ───────────────────────────────────────────────

def check_guardrails(state: AgentState) -> dict:
    """
    Validate the agent's response before sending to patient.

    Uses the GuardrailEngine (Project 7) which runs 7 checks:
    1. Service hallucination detection
    2. Never-say phrase filtering
    3. Off-limits topic blocking
    4. Conversation ending detection
    5. Confidence-based assertion gating
    6. Response quality validation
    7. PII leak prevention

    If a violation is found, flags for regeneration or blocks delivery.
    """
    messages = state["messages"]
    if not messages:
        return {"guardrail_violation": False, "last_node": "check_guardrails"}

    last_message = messages[-1]

    # Only check AI messages that have text content (not tool calls)
    if not isinstance(last_message, AIMessage) or not last_message.content:
        return {"guardrail_violation": False, "last_node": "check_guardrails"}

    if last_message.tool_calls:
        return {"guardrail_violation": False, "last_node": "check_guardrails"}

    from app.guardrails import run_guardrails

    result = run_guardrails(
        response=last_message.content,
        practice_id=state["practice_id"],
        intent=state.get("intent", ""),
        confidence=state.get("confidence", 1.0),
        booking_confirmed=state.get("booking_confirmed", False),
    )

    if result.blocked or result.regenerate:
        logger.warning(f"Guardrail violations: {result.violations}")
        return {
            "guardrail_violation": True,
            "last_node": "check_guardrails",
        }

    # If response was modified (e.g., truncated), replace the message
    if result.modified_response:
        return {
            "messages": [AIMessage(content=result.modified_response)],
            "guardrail_violation": False,
            "last_node": "check_guardrails",
        }

    return {"guardrail_violation": False, "last_node": "check_guardrails"}


# ── Node 5: Handle Escalation ─────────────────────────────────────────────

def handle_escalation(state: AgentState) -> dict:
    """
    Handle escalation to human staff.
    Sends the practice's configured escalation message and flags the conversation.
    """
    practice = get_practice(state["practice_id"])
    escalation_msg = practice["escalation"]["escalation_message"]

    return {
        "messages": [AIMessage(content=escalation_msg)],
        "needs_escalation": True,
        "last_node": "handle_escalation",
    }


# ── Node 6: Handle Greeting ───────────────────────────────────────────────

def handle_greeting(state: AgentState) -> dict:
    """Respond to initial greetings with the practice's configured greeting."""
    practice = get_practice(state["practice_id"])
    return {
        "messages": [AIMessage(content=practice["greeting"])],
        "last_node": "handle_greeting",
    }


# ── Node 7: Handle Farewell ───────────────────────────────────────────────

def handle_farewell(state: AgentState) -> dict:
    """
    Handle conversation endings gracefully.
    Agents commonly struggle with not detecting when conversations have naturally ended,
    leading to awkward repeated goodbyes.
    """
    if state.get("booking_confirmed"):
        msg = f"You're all set! We look forward to seeing you. If anything changes, just text us here. Have a great day! 😊"
    else:
        msg = "Thank you for reaching out! If you need anything else, just text us anytime. Have a great day!"

    return {
        "messages": [AIMessage(content=msg)],
        "conversation_ended": True,
        "last_node": "handle_farewell",
    }
