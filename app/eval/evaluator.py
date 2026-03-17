"""
Agent Evaluation Module — LangSmith integration for Project 2.

Provides:
- Baseline dataset creation from test conversations
- LLM-as-judge evaluator for response quality
- Pairwise comparison runner
- F1 score tracking for intent classification

This implements a production evaluation workflow:
  Curate dataset → Offline eval → Ship → Monitor → Expand dataset → Repeat
"""
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Test Scenarios for Baseline Dataset ───────────────────────────────────

BASELINE_SCENARIOS = [
    # --- Booking Flows ---
    {
        "id": "book-001",
        "category": "booking",
        "input": "I need a teeth cleaning next Tuesday",
        "expected_intent": "book_appointment",
        "expected_behavior": "Should check availability for General Cleaning on next Tuesday",
        "expected_tools": ["check_availability"],
    },
    {
        "id": "book-002",
        "category": "booking",
        "input": "Can I get a dental exam this week?",
        "expected_intent": "book_appointment",
        "expected_behavior": "Should check availability for Dental Exam",
        "expected_tools": ["check_availability"],
    },
    {
        "id": "book-003",
        "category": "booking",
        "input": "I want to whiten my teeth",
        "expected_intent": "book_appointment",
        "expected_behavior": "Should identify Teeth Whitening service and offer to book",
        "expected_tools": ["lookup_service_info"],
    },
    {
        "id": "book-004",
        "category": "booking",
        "input": "My tooth really hurts, can I come in today?",
        "expected_intent": "book_appointment",
        "expected_behavior": "Should recognize emergency, offer same-day Emergency Visit",
        "expected_tools": ["check_availability"],
    },

    # --- Mid-Flow Changes ---
    {
        "id": "change-001",
        "category": "mid_flow",
        "input": "Actually, can we do Wednesday instead?",
        "expected_intent": "reschedule",
        "expected_behavior": "Should check availability for Wednesday, same service",
        "expected_tools": ["check_availability"],
    },
    {
        "id": "change-002",
        "category": "mid_flow",
        "input": "nvm",
        "expected_intent": "cancel",
        "expected_behavior": "Should acknowledge cancellation gracefully, offer to help with anything else",
        "expected_tools": [],
    },

    # --- Service Questions ---
    {
        "id": "question-001",
        "category": "question",
        "input": "How long does a cleaning take?",
        "expected_intent": "ask_question",
        "expected_behavior": "Should answer: approximately 60 minutes",
        "expected_tools": ["lookup_service_info"],
    },
    {
        "id": "question-002",
        "category": "question",
        "input": "Do you guys do veneers?",
        "expected_intent": "ask_question",
        "expected_behavior": "Should say veneers are not on the service list, offer to check with team",
        "expected_tools": [],
    },
    {
        "id": "question-003",
        "category": "question",
        "input": "What should I do to prepare for a root canal?",
        "expected_intent": "ask_question",
        "expected_behavior": "Should provide preparation info from knowledge base",
        "expected_tools": ["lookup_service_info"],
    },

    # --- Insurance Questions ---
    {
        "id": "insurance-001",
        "category": "insurance",
        "input": "Do you take Delta Dental?",
        "expected_intent": "insurance_question",
        "expected_behavior": "Should confirm Delta Dental is accepted",
        "expected_tools": ["check_insurance"],
    },
    {
        "id": "insurance-002",
        "category": "insurance",
        "input": "Is Humana accepted?",
        "expected_intent": "insurance_question",
        "expected_behavior": "Should say not sure, offer to have billing team verify",
        "expected_tools": ["check_insurance"],
    },

    # --- Edge Cases ---
    {
        "id": "edge-001",
        "category": "edge_case",
        "input": "Do you guys do that thing... you know... the face thing",
        "expected_intent": "unclear",
        "expected_behavior": "Should ask clarifying question. Should NOT guess a service.",
        "expected_tools": [],
    },
    {
        "id": "edge-002",
        "category": "edge_case",
        "input": "k",
        "expected_intent": "unclear",
        "expected_behavior": "Should interpret in context. If mid-booking, treat as acknowledgment.",
        "expected_tools": [],
    },
    {
        "id": "edge-003",
        "category": "edge_case",
        "input": "Necesito una limpieza dental",
        "expected_intent": "book_appointment",
        "expected_behavior": "Should understand Spanish request for dental cleaning and respond helpfully",
        "expected_tools": [],
    },
    {
        "id": "edge-004",
        "category": "edge_case",
        "input": "How much is a cleaning?",
        "expected_intent": "ask_question",
        "expected_behavior": "Should NOT give specific pricing. Should say pricing varies and offer to discuss at visit.",
        "expected_tools": [],
    },

    # --- Escalation Triggers ---
    {
        "id": "escalate-001",
        "category": "escalation",
        "input": "I want to talk to a real person",
        "expected_intent": "escalate_to_human",
        "expected_behavior": "Should immediately escalate. Do NOT try to keep the patient.",
        "expected_tools": ["escalate_to_human"],
    },
    {
        "id": "escalate-002",
        "category": "escalation",
        "input": "I'm having severe chest pain and my jaw hurts",
        "expected_intent": "escalate_to_human",
        "expected_behavior": "Should escalate immediately. This could be a medical emergency.",
        "expected_tools": ["escalate_to_human"],
    },
    {
        "id": "escalate-003",
        "category": "escalation",
        "input": "I want to file a complaint about my last visit",
        "expected_intent": "escalate_to_human",
        "expected_behavior": "Should escalate to human team with empathy",
        "expected_tools": ["escalate_to_human"],
    },

    # --- Greetings & Farewells ---
    {
        "id": "greeting-001",
        "category": "greeting",
        "input": "Hi",
        "expected_intent": "greeting",
        "expected_behavior": "Should respond with practice greeting",
        "expected_tools": [],
    },
    {
        "id": "farewell-001",
        "category": "farewell",
        "input": "Thanks, that's all I needed!",
        "expected_intent": "farewell",
        "expected_behavior": "Should send brief closing message and STOP. No follow-up questions.",
        "expected_tools": [],
    },
    {
        "id": "farewell-002",
        "category": "farewell",
        "input": "See you Tuesday!",
        "expected_intent": "farewell",
        "expected_behavior": "Should confirm and end. ONE response only.",
        "expected_tools": [],
    },

    # --- Guardrail Tests ---
    {
        "id": "guard-001",
        "category": "guardrail",
        "input": "Can you do Botox?",
        "expected_intent": "ask_question",
        "expected_behavior": "For dental practice: should say NOT offered. Should NOT hallucinate that they do Botox.",
        "expected_tools": [],
    },
    {
        "id": "guard-002",
        "category": "guardrail",
        "input": "Are you a real person or a bot?",
        "expected_intent": "ask_question",
        "expected_behavior": "Should NOT say 'I'm an AI' or 'I'm a chatbot'. Should redirect to helping.",
        "expected_tools": [],
    },
]


def get_baseline_scenarios(category: str = None) -> list[dict]:
    """Get baseline test scenarios, optionally filtered by category."""
    if category:
        return [s for s in BASELINE_SCENARIOS if s["category"] == category]
    return BASELINE_SCENARIOS


def evaluate_intent_accuracy(results: list[dict]) -> dict:
    """
    Calculate intent classification accuracy metrics.

    Args:
        results: List of dicts with 'expected_intent' and 'actual_intent'

    Returns:
        Dict with accuracy, per-intent precision/recall, and F1 score
    """
    total = len(results)
    correct = sum(1 for r in results if r["expected_intent"] == r["actual_intent"])

    # Per-intent metrics
    intents = set(r["expected_intent"] for r in results)
    per_intent = {}

    for intent in intents:
        tp = sum(1 for r in results if r["expected_intent"] == intent and r["actual_intent"] == intent)
        fp = sum(1 for r in results if r["expected_intent"] != intent and r["actual_intent"] == intent)
        fn = sum(1 for r in results if r["expected_intent"] == intent and r["actual_intent"] != intent)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_intent[intent] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for r in results if r["expected_intent"] == intent),
        }

    # Macro F1 (average across intents)
    macro_f1 = sum(m["f1"] for m in per_intent.values()) / len(per_intent) if per_intent else 0

    return {
        "accuracy": round(correct / total, 3) if total > 0 else 0,
        "total": total,
        "correct": correct,
        "macro_f1": round(macro_f1, 3),
        "per_intent": per_intent,
        "evaluated_at": datetime.now().isoformat(),
    }
