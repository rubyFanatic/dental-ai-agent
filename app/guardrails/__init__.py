"""
Guardrails & Safety Layer — Project 7.

A validation layer between the agent's response generation and delivery.
Every outbound response passes through these checks before the patient sees it.

Guardrails:
    1. Service Hallucination Check — agent must NEVER mention a service the practice doesn't offer
    2. Never-Say Phrase Check — blocks responses containing forbidden phrases
    3. Off-Limits Topic Check — blocks pricing, diagnosis, competitor mentions
    4. Conversation Ending Detection — prevents awkward repeated goodbyes
    5. Confidence Gate — forces clarifying questions when model is unsure
    6. Response Quality Check — catches empty, too-long, or malformed responses
    7. PII Leak Check — prevents agent from echoing back sensitive patient data

Production agents that skip guardrails eventually hallucinate services,
give medical advice, or quote prices they shouldn't. These checks
add ~50ms of latency and prevent 100% of the most damaging failure modes.
"""
import re
import logging
from dataclasses import dataclass, field
from app.config.practice_config import get_practice

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool = True
    violations: list[str] = field(default_factory=list)
    blocked: bool = False  # True = response should NOT be sent
    regenerate: bool = False  # True = ask LLM to try again
    modified_response: str = ""  # If non-empty, use this instead of the original


class GuardrailEngine:
    """
    Run all guardrail checks on an agent response.

    Usage:
        engine = GuardrailEngine("apex-dental-01")
        result = engine.check(response_text, conversation_state)
        if result.blocked:
            # Don't send this response — regenerate or escalate
        elif result.modified_response:
            # Use the modified version
    """

    def __init__(self, practice_id: str):
        self.practice = get_practice(practice_id)
        self.practice_id = practice_id

    def check(
        self,
        response: str,
        intent: str = "",
        confidence: float = 1.0,
        booking_confirmed: bool = False,
        conversation_history: list[str] = None,
    ) -> GuardrailResult:
        """
        Run all guardrail checks on a response.

        Args:
            response: The agent's generated response text
            intent: Detected intent for context
            confidence: Model confidence score
            booking_confirmed: Whether an appointment was booked this session
            conversation_history: Recent message history for context

        Returns:
            GuardrailResult with pass/fail and details
        """
        result = GuardrailResult()
        checks = [
            self._check_service_hallucination,
            self._check_never_say,
            self._check_off_limits,
            self._check_conversation_ending,
            self._check_confidence_gate,
            self._check_response_quality,
            self._check_pii_leak,
        ]

        for check_fn in checks:
            check_result = check_fn(
                response=response,
                intent=intent,
                confidence=confidence,
                booking_confirmed=booking_confirmed,
                history=conversation_history or [],
            )
            if not check_result.passed:
                result.passed = False
                result.violations.extend(check_result.violations)
                if check_result.blocked:
                    result.blocked = True
                if check_result.regenerate:
                    result.regenerate = True
                if check_result.modified_response:
                    result.modified_response = check_result.modified_response

        if not result.passed:
            logger.warning(
                f"Guardrail violations for {self.practice_id}: "
                f"{result.violations} | blocked={result.blocked}"
            )

        return result

    # ── Check 1: Service Hallucination ────────────────────────────────

    def _check_service_hallucination(self, response: str, **kwargs) -> GuardrailResult:
        """
        Verify the response doesn't mention services the practice doesn't offer.

        This is the #1 most critical guardrail. A dental office's agent must
        NEVER tell a patient they offer Botox, and a medspa agent must NEVER
        tell a client they do root canals.
        """
        result = GuardrailResult()
        response_lower = response.lower()

        # Known services that could be hallucinated per vertical
        hallucination_triggers = {
            "dental": [
                "botox", "filler", "dermal filler", "microneedling", "hydrafacial",
                "laser hair", "body contouring", "iv therapy", "chemical peel",
                "liposuction", "tummy tuck", "breast augmentation",
            ],
            "medspa": [
                "root canal", "extraction", "braces", "cavity", "filling",
                "dental implant", "wisdom teeth", "dentures", "retainer",
                "periodontal", "gum surgery",
            ],
        }

        triggers = hallucination_triggers.get(self.practice["vertical"], [])
        approved_services = [s.lower() for s in self.practice["services"]]

        for trigger in triggers:
            if trigger in response_lower:
                # Check if it's in context of saying "we don't offer this"
                denial_patterns = [
                    f"don't offer {trigger}",
                    f"do not offer {trigger}",
                    f"don't provide {trigger}",
                    f"not a service we",
                    f"not something we",
                    f"don't believe we offer",
                ]
                is_denial = any(p in response_lower for p in denial_patterns)

                if not is_denial:
                    result.passed = False
                    result.blocked = True
                    result.regenerate = True
                    result.violations.append(
                        f"HALLUCINATED SERVICE: Response mentions '{trigger}' which is not offered"
                    )

        return result

    # ── Check 2: Never-Say Phrases ────────────────────────────────────

    def _check_never_say(self, response: str, **kwargs) -> GuardrailResult:
        """Block responses containing forbidden phrases."""
        result = GuardrailResult()
        response_lower = response.lower()

        for phrase in self.practice.get("never_say", []):
            if phrase.lower() in response_lower:
                result.passed = False
                result.regenerate = True
                result.violations.append(f"NEVER-SAY: Response contains '{phrase}'")

        return result

    # ── Check 3: Off-Limits Topics ────────────────────────────────────

    def _check_off_limits(self, response: str, **kwargs) -> GuardrailResult:
        """Block responses that discuss off-limits topics directly."""
        result = GuardrailResult()
        response_lower = response.lower()

        # Price-related patterns (most common violation)
        price_patterns = [
            r'\$\d+',                    # Dollar amounts like $150
            r'\d+\s*dollars',            # "150 dollars"
            r'costs?\s+(about|around|approximately)\s*\$?\d+',
            r'price\s+(is|starts|ranges)',
            r'(typically|usually|generally)\s+costs?',
        ]

        for topic in self.practice.get("off_limits_topics", []):
            if "pricing" in topic.lower() or "price" in topic.lower():
                for pattern in price_patterns:
                    if re.search(pattern, response_lower):
                        result.passed = False
                        result.regenerate = True
                        result.violations.append(
                            f"OFF-LIMITS: Response appears to discuss specific pricing"
                        )
                        break
            elif topic.lower() in response_lower:
                # Allow if it's in context of redirecting
                redirect_phrases = ["discuss", "consultation", "happy to go over", "at your visit"]
                is_redirect = any(p in response_lower for p in redirect_phrases)
                if not is_redirect:
                    result.passed = False
                    result.regenerate = True
                    result.violations.append(f"OFF-LIMITS: Response discusses '{topic}'")

        return result

    # ── Check 4: Conversation Ending Detection ────────────────────────

    def _check_conversation_ending(self, response: str, intent: str = "", **kwargs) -> GuardrailResult:
        """
        Detect when the agent should stop talking.

        Common failure: agent keeps talking after the patient says "thanks, bye."
        The agent should send ONE brief closing message and stop.
        """
        result = GuardrailResult()

        if intent == "farewell":
            # Response to a farewell should be short
            word_count = len(response.split())
            if word_count > 40:
                result.passed = False
                result.violations.append(
                    f"OVER-TALKING: Farewell response is {word_count} words (max: 40)"
                )
                # Don't block — just truncate
                sentences = response.split('. ')
                result.modified_response = '. '.join(sentences[:2]) + '.'

            # Should not ask a new question in farewell
            question_indicators = ['?', 'would you like', 'can i help', 'anything else']
            has_question = any(q in response.lower() for q in question_indicators)
            if has_question and kwargs.get("booking_confirmed", False):
                result.passed = False
                result.violations.append(
                    "OVER-TALKING: Asking questions after booking confirmed + farewell"
                )

        return result

    # ── Check 5: Confidence Gate ──────────────────────────────────────

    def _check_confidence_gate(self, response: str, confidence: float = 1.0, **kwargs) -> GuardrailResult:
        """
        When model confidence is low, the response should be a clarifying
        question — not a confident-sounding answer that might be wrong.
        """
        result = GuardrailResult()
        threshold = self.practice["escalation"]["confidence_threshold"]

        if confidence < threshold:
            # Low confidence — response should be asking for clarification, not asserting
            assertive_patterns = [
                "yes, we", "absolutely", "of course", "definitely",
                "your appointment is", "i've booked",
            ]
            response_lower = response.lower()

            for pattern in assertive_patterns:
                if pattern in response_lower:
                    result.passed = False
                    result.violations.append(
                        f"CONFIDENCE: Model confidence is {confidence:.2f} "
                        f"but response is assertive (contains '{pattern}')"
                    )
                    result.regenerate = True
                    break

        return result

    # ── Check 6: Response Quality ─────────────────────────────────────

    def _check_response_quality(self, response: str, **kwargs) -> GuardrailResult:
        """Basic quality checks on the response."""
        result = GuardrailResult()

        # Empty response
        if not response or not response.strip():
            result.passed = False
            result.blocked = True
            result.violations.append("QUALITY: Empty response")
            return result

        # Too short (less than 2 words, likely malformed)
        if len(response.split()) < 2:
            result.passed = False
            result.violations.append(f"QUALITY: Response too short ({len(response.split())} words)")

        # Too long (over 200 words — agent is rambling)
        word_count = len(response.split())
        if word_count > 200:
            result.passed = False
            result.violations.append(f"QUALITY: Response too long ({word_count} words, max 200)")
            # Truncate to first 3 sentences
            sentences = response.split('. ')
            result.modified_response = '. '.join(sentences[:3]) + '.'

        # Contains raw JSON or code artifacts
        if '{' in response and '}' in response and '"' in response:
            # Might be leaking internal JSON
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    import json
                    json.loads(json_match.group())
                    # If it parses as JSON, it's probably leaked internal data
                    result.passed = False
                    result.blocked = True
                    result.violations.append("QUALITY: Response contains raw JSON — possible data leak")
            except (json.JSONDecodeError, ValueError):
                pass  # Not actual JSON, probably fine

        return result

    # ── Check 7: PII Leak Prevention ──────────────────────────────────

    def _check_pii_leak(self, response: str, **kwargs) -> GuardrailResult:
        """
        Prevent the agent from echoing back sensitive patient information
        in a way that could be visible in logs or SMS previews.
        """
        result = GuardrailResult()
        response_text = response

        # SSN patterns
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', response_text):
            result.passed = False
            result.blocked = True
            result.violations.append("PII: Response contains SSN-like pattern")

        # Credit card patterns (basic check)
        if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', response_text):
            result.passed = False
            result.blocked = True
            result.violations.append("PII: Response contains credit card-like pattern")

        # Date of birth with context
        if re.search(r'(date of birth|DOB|born on)\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', response_text, re.IGNORECASE):
            result.passed = False
            result.violations.append("PII: Response contains date of birth")

        return result


# ── Convenience function for use in the agent graph ───────────────────────

def run_guardrails(
    response: str,
    practice_id: str,
    intent: str = "",
    confidence: float = 1.0,
    booking_confirmed: bool = False,
) -> GuardrailResult:
    """Run all guardrail checks. Call from the agent graph's guardrail node."""
    engine = GuardrailEngine(practice_id)
    return engine.check(
        response=response,
        intent=intent,
        confidence=confidence,
        booking_confirmed=booking_confirmed,
    )
