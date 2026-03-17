"""
Test the guardrail engine against known good and bad responses.

Usage:
    python scripts/test_guardrails.py               # Run all tests
    python scripts/test_guardrails.py --verbose      # Show details for each test

No API keys needed — guardrails run entirely locally.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.guardrails import GuardrailEngine


# ── Test Cases ────────────────────────────────────────────────────────────

TESTS = [
    # === Should PASS ===
    {
        "name": "Clean booking response",
        "response": "I'd be happy to help you schedule a General Cleaning! We have openings on Tuesday at 10am and Thursday at 2pm. Which works better for you?",
        "practice_id": "apex-dental-01",
        "should_pass": True,
    },
    {
        "name": "Clean farewell (short)",
        "response": "You're all set! We look forward to seeing you Tuesday. Have a great day!",
        "practice_id": "apex-dental-01",
        "intent": "farewell",
        "booking_confirmed": True,
        "should_pass": True,
    },
    {
        "name": "Redirects pricing question",
        "response": "Pricing depends on your specific situation. We'd be happy to discuss that at your visit, or I can have our billing team give you a call.",
        "practice_id": "apex-dental-01",
        "should_pass": True,
    },
    {
        "name": "Properly says service not offered",
        "response": "I don't believe we offer Botox at our practice, but let me check with our team. Can I have someone call you?",
        "practice_id": "apex-dental-01",
        "should_pass": True,
    },

    # === Should FAIL: Service Hallucination ===
    {
        "name": "Hallucinated Botox at dental office",
        "response": "Yes! We offer Botox treatments. Would you like to schedule an appointment?",
        "practice_id": "apex-dental-01",
        "should_pass": False,
        "expected_violation": "HALLUCINATED SERVICE",
    },
    {
        "name": "Hallucinated root canal at medspa",
        "response": "We can definitely help with that root canal. Let me check availability.",
        "practice_id": "cary-medspa-01",
        "should_pass": False,
        "expected_violation": "HALLUCINATED SERVICE",
    },

    # === Should FAIL: Never-Say ===
    {
        "name": "Says 'I'm an AI'",
        "response": "I'm an AI assistant here to help you. What can I do for you?",
        "practice_id": "apex-dental-01",
        "should_pass": False,
        "expected_violation": "NEVER-SAY",
    },
    {
        "name": "Says 'I don't know'",
        "response": "I don't know the answer to that. Can you ask again?",
        "practice_id": "apex-dental-01",
        "should_pass": False,
        "expected_violation": "NEVER-SAY",
    },
    {
        "name": "Medspa says 'injectable'",
        "response": "Our injectable treatments are very popular! Would you like to learn more?",
        "practice_id": "cary-medspa-01",
        "should_pass": False,
        "expected_violation": "NEVER-SAY",
    },

    # === Should FAIL: Off-Limits (Pricing) ===
    {
        "name": "Gives specific price",
        "response": "A general cleaning costs $150 without insurance.",
        "practice_id": "apex-dental-01",
        "should_pass": False,
        "expected_violation": "OFF-LIMITS",
    },
    {
        "name": "Gives dollar amount",
        "response": "That treatment typically costs around $250.",
        "practice_id": "apex-dental-01",
        "should_pass": False,
        "expected_violation": "OFF-LIMITS",
    },

    # === Should FAIL: Response Quality ===
    {
        "name": "Empty response",
        "response": "",
        "practice_id": "apex-dental-01",
        "should_pass": False,
        "expected_violation": "QUALITY",
    },
    {
        "name": "Single word response",
        "response": "Yes",
        "practice_id": "apex-dental-01",
        "should_pass": False,
        "expected_violation": "QUALITY",
    },

    # === Should FAIL: Confidence Gate ===
    {
        "name": "Assertive response at low confidence",
        "response": "Yes, we definitely offer that service! Let me book you in.",
        "practice_id": "apex-dental-01",
        "confidence": 0.3,
        "should_pass": False,
        "expected_violation": "CONFIDENCE",
    },

    # === Should FAIL: PII Leak ===
    {
        "name": "Echoes SSN pattern",
        "response": "I have your SSN as 123-45-6789. Is that correct?",
        "practice_id": "apex-dental-01",
        "should_pass": False,
        "expected_violation": "PII",
    },

    # === Should FAIL: Over-talking on farewell ===
    {
        "name": "Long farewell with questions after booking",
        "response": "Thank you so much for choosing us! We really appreciate your business. Your appointment is confirmed for Tuesday at 10am for a General Cleaning. Please arrive 15 minutes early to complete paperwork. Bring your insurance card and photo ID. Is there anything else I can help you with today? Would you also like to schedule a follow-up whitening appointment?",
        "practice_id": "apex-dental-01",
        "intent": "farewell",
        "booking_confirmed": True,
        "should_pass": False,
        "expected_violation": "OVER-TALKING",
    },
]


def run_tests(verbose: bool = False):
    """Run all guardrail tests."""
    passed = 0
    failed = 0
    errors = []

    print("\n" + "=" * 60)
    print("  GUARDRAIL ENGINE TESTS")
    print("=" * 60 + "\n")

    for test in TESTS:
        engine = GuardrailEngine(test["practice_id"])
        result = engine.check(
            response=test["response"],
            intent=test.get("intent", ""),
            confidence=test.get("confidence", 1.0),
            booking_confirmed=test.get("booking_confirmed", False),
        )

        test_passed = result.passed == test["should_pass"]

        if test_passed:
            passed += 1
            icon = "\033[92m\u2713\033[0m"  # Green check
        else:
            failed += 1
            icon = "\033[91m\u2717\033[0m"  # Red X
            errors.append(test["name"])

        print(f"  {icon}  {test['name']}")

        if verbose or not test_passed:
            expected = "PASS" if test["should_pass"] else "FAIL"
            actual = "PASS" if result.passed else "FAIL"
            print(f"       Expected: {expected} | Actual: {actual}")
            if result.violations:
                for v in result.violations:
                    print(f"       Violation: {v}")
            if result.modified_response:
                print(f"       Modified: {result.modified_response[:60]}...")
            print()

    print(f"\n  Results: {passed} passed, {failed} failed, {len(TESTS)} total")
    if errors:
        print(f"\n  Failed tests:")
        for e in errors:
            print(f"    - {e}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    success = run_tests(verbose)
    sys.exit(0 if success else 1)
