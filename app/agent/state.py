"""
LangGraph Conversation State — persists across turns via checkpointer.

This is the core data structure that flows through every node in the graph.
Each node reads what it needs, does its work, and returns updated fields.
LangGraph merges the updates automatically.

The `messages` field uses LangGraph's built-in message handling with
the `add_messages` reducer, which appends new messages rather than replacing.
"""
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Full conversation state for the dental booking agent.

    This state travels through every node in the LangGraph.
    Nodes read what they need and return only the fields they update.
    """

    # ── Conversation History ──────────────────────────────────────────
    # Uses add_messages reducer: new messages are APPENDED, not replaced.
    # This is how multi-turn context is maintained.
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Practice Context ──────────────────────────────────────────────
    practice_id: str           # Which practice this conversation is for
    channel: str               # "sms", "webchat", or "voice"

    # ── Intent & Entities (set by intent classifier) ──────────────────
    intent: str                # Detected intent: book, reschedule, cancel, question, escalate, greeting, farewell
    confidence: float          # Model's confidence in the classification (0.0 - 1.0)
    extracted_service: str     # Service name extracted from message (if any)
    extracted_date: str        # Date/time preference extracted (if any)
    extracted_name: str        # Patient name if provided

    # ── Service Lookup (set by service lookup node) ────────────────────
    matched_service: str       # Verified service from practice catalog
    service_details: str       # Description, duration, prep instructions from RAG

    # ── Availability (set by availability check node) ──────────────────
    available_slots: list[dict]  # Available time slots from calendar
    selected_slot: dict        # The slot the patient chose

    # ── Booking (set by booking node) ──────────────────────────────────
    booking_confirmed: bool    # Whether appointment is booked
    booking_id: str            # Booking reference ID

    # ── Routing Flags ─────────────────────────────────────────────────
    needs_escalation: bool     # True = route to human
    escalation_reason: str     # Why escalation was triggered
    conversation_ended: bool   # True = natural end, stop processing
    guardrail_violation: bool  # True = response blocked, regenerate

    # ── Metadata ──────────────────────────────────────────────────────
    turn_count: int            # Number of conversation turns
    last_node: str             # Which node last processed this state
