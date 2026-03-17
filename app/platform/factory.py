"""
Multi-Vertical Platform Factory — Project 6.

The platform pattern: same agent engine serves multiple verticals
(dental, medspa, PT clinic, HVAC, auto). Adding a new vertical is
a configuration change, not a code change.

What's per-vertical:
    - System prompt template (tone, rules, terminology)
    - Service catalog
    - Booking rules (consultation requirements, notice periods)
    - Guardrail config (never-say lists, off-limits topics)
    - Pinecone namespace (knowledge base)

What's shared (the platform):
    - LangGraph agent flow
    - Tool definitions and implementations
    - Evaluation pipeline
    - Observability metrics
    - Channel adapters (SMS, voice, web chat)

Usage:
    from app.platform.factory import PlatformFactory

    factory = PlatformFactory()
    agent = factory.create_agent("apex-dental-01")
    medspa_agent = factory.create_agent("cary-medspa-01")
    # Same engine, different config. Zero code changes.

    # Register a brand new vertical
    factory.register_vertical("pt_clinic", PT_CLINIC_TEMPLATE)
"""
import logging
from copy import deepcopy
from langgraph.checkpoint.memory import MemorySaver
from app.agent.graph import build_agent
from app.agent.prompts import build_system_prompt
from app.config.practice_config import get_practice, PRACTICES

logger = logging.getLogger(__name__)


# ── Vertical Templates ────────────────────────────────────────────────────
# Each vertical has a base template. Individual practices customize from here.

VERTICAL_TEMPLATES = {
    "dental": {
        "tone": "friendly, professional, caring",
        "terminology": {
            "provider": "dentist",
            "client": "patient",
            "visit": "appointment",
            "location": "office",
        },
        "default_greeting": "Hi! Thanks for reaching out. How can we help you today?",
        "insurance_supported": True,
        "requires_consultation_for": [],
        "emergency_same_day": True,
        "off_limits_topics": ["specific pricing", "medical diagnosis", "competitor practices"],
        "never_say": ["I'm an AI", "I'm a chatbot", "I don't know"],
    },

    "medspa": {
        "tone": "luxury, warm, empowering, professional",
        "terminology": {
            "provider": "specialist",
            "client": "client",
            "visit": "appointment",
            "location": "studio",
        },
        "default_greeting": "Hi! Welcome to our studio. We'd love to help you look and feel your best. What can I help you with?",
        "insurance_supported": False,
        "requires_consultation_for": ["Botox", "Dermal Fillers", "Body Contouring"],
        "emergency_same_day": False,
        "off_limits_topics": ["specific pricing over text", "medical diagnosis", "guarantee results", "competitor practices"],
        "never_say": ["I'm an AI", "I'm a chatbot", "cheap", "discount", "injectable"],
    },

    "pt_clinic": {
        "tone": "encouraging, professional, knowledgeable",
        "terminology": {
            "provider": "therapist",
            "client": "patient",
            "visit": "session",
            "location": "clinic",
        },
        "default_greeting": "Hi! Thanks for reaching out. We're here to help you get back to feeling your best. How can I help?",
        "insurance_supported": True,
        "requires_consultation_for": ["Initial Evaluation"],
        "emergency_same_day": False,
        "off_limits_topics": ["specific diagnosis", "treatment guarantees", "competitor practices"],
        "never_say": ["I'm an AI", "I'm a chatbot", "I don't know"],
    },

    "hvac": {
        "tone": "friendly, reliable, straightforward",
        "terminology": {
            "provider": "technician",
            "client": "customer",
            "visit": "service call",
            "location": "shop",
        },
        "default_greeting": "Hey! Thanks for reaching out. How can we help you today?",
        "insurance_supported": False,
        "requires_consultation_for": [],
        "emergency_same_day": True,
        "off_limits_topics": ["competitor pricing", "DIY repair advice for gas systems"],
        "never_say": ["I'm an AI", "I'm a chatbot"],
    },
}


class PlatformFactory:
    """
    Factory for creating per-practice agent instances from shared platform components.

    The factory pattern ensures:
    1. Every practice gets the same battle-tested agent engine
    2. Vertical-specific behavior comes from configuration, not code
    3. New verticals can be registered at runtime
    4. Each practice has isolated conversation state
    """

    def __init__(self):
        self._verticals = deepcopy(VERTICAL_TEMPLATES)
        self._agents = {}  # Cache of compiled agents per practice
        logger.info(f"Platform initialized with {len(self._verticals)} verticals: "
                    f"{list(self._verticals.keys())}")

    def register_vertical(self, vertical_name: str, template: dict):
        """
        Register a new vertical template.

        This is how you add support for a new business type
        without changing any agent code.

        Args:
            vertical_name: e.g., "auto_dealer", "jewelry_store"
            template: Dict matching the VERTICAL_TEMPLATES structure
        """
        required_keys = ["tone", "terminology", "default_greeting"]
        for key in required_keys:
            if key not in template:
                raise ValueError(f"Vertical template missing required key: {key}")

        self._verticals[vertical_name] = template
        logger.info(f"Registered new vertical: {vertical_name}")

    def get_verticals(self) -> list[str]:
        """List all registered verticals."""
        return list(self._verticals.keys())

    def get_vertical_template(self, vertical: str) -> dict:
        """Get the template for a vertical."""
        if vertical not in self._verticals:
            raise KeyError(f"Unknown vertical: {vertical}. Available: {self.get_verticals()}")
        return deepcopy(self._verticals[vertical])

    def create_agent(self, practice_id: str, checkpointer=None):
        """
        Create a compiled LangGraph agent for a specific practice.

        The agent uses the shared graph (same nodes, same edges) but
        with practice-specific configuration injected at runtime.

        Args:
            practice_id: Practice to create the agent for
            checkpointer: State persistence backend (defaults to MemorySaver)

        Returns:
            Compiled LangGraph application
        """
        # Validate practice exists
        practice = get_practice(practice_id)
        vertical = practice["vertical"]

        if vertical not in self._verticals:
            raise ValueError(
                f"Practice '{practice_id}' uses vertical '{vertical}' "
                f"which is not registered. Available: {self.get_verticals()}"
            )

        # Build agent with practice-specific checkpointer
        if checkpointer is None:
            checkpointer = MemorySaver()

        agent = build_agent(checkpointer=checkpointer)

        logger.info(f"Created agent for {practice['name']} ({vertical})")
        return agent

    def create_all_agents(self) -> dict:
        """
        Create agents for all registered practices.

        Returns:
            Dict mapping practice_id → compiled agent
        """
        agents = {}
        for practice_id in PRACTICES:
            try:
                agents[practice_id] = self.create_agent(practice_id)
            except Exception as e:
                logger.error(f"Failed to create agent for {practice_id}: {e}")

        logger.info(f"Created {len(agents)} agents for {len(PRACTICES)} practices")
        return agents

    def validate_practice_config(self, practice_id: str) -> dict:
        """
        Validate a practice configuration against its vertical template.

        Returns a dict with validation status and any warnings.
        Useful for onboarding new practices.
        """
        practice = get_practice(practice_id)
        vertical = practice["vertical"]
        template = self.get_vertical_template(vertical)

        issues = []
        warnings = []

        # Check required fields
        required = ["services", "hours", "booking_rules", "escalation", "tone"]
        for field in required:
            if field not in practice:
                issues.append(f"Missing required field: {field}")

        # Check services exist
        if not practice.get("services"):
            issues.append("No services defined — agent will have nothing to book")

        # Check booking rules have durations for all services
        durations = practice.get("booking_rules", {}).get("slot_duration_minutes", {})
        for service in practice.get("services", []):
            if service not in durations:
                warnings.append(f"No duration defined for '{service}' — will default to 30 min")

        # Check insurance config matches vertical
        if template.get("insurance_supported") and not practice.get("insurance"):
            warnings.append(f"Vertical '{vertical}' supports insurance but no plans listed")

        # Check escalation config
        escalation = practice.get("escalation", {})
        if "confidence_threshold" not in escalation:
            warnings.append("No confidence threshold set — will default to 0.7")

        # Check never_say list
        if not practice.get("never_say"):
            warnings.append("No 'never_say' list — agent has no phrase guardrails")

        return {
            "practice_id": practice_id,
            "vertical": vertical,
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "services_count": len(practice.get("services", [])),
            "has_insurance": bool(practice.get("insurance")),
            "has_escalation": bool(escalation),
        }


# ── Practice Onboarding Helper ────────────────────────────────────────────

def onboard_new_practice(
    practice_id: str,
    practice_name: str,
    vertical: str,
    services: list[str],
    hours: dict,
    phone: str = "",
    address: str = "",
    insurance: list[str] = None,
    custom_greeting: str = None,
    custom_tone: str = None,
    never_say: list[str] = None,
) -> dict:
    """
    Generate a complete practice configuration from minimal inputs.

    This is the onboarding path for new practices:
    1. Collect basic info (name, vertical, services, hours)
    2. Apply vertical template defaults
    3. Generate full config
    4. Validate
    5. Add to PRACTICES dict

    Returns the generated config dict.
    """
    factory = PlatformFactory()
    template = factory.get_vertical_template(vertical)

    # Build config with template defaults
    config = {
        "practice_id": practice_id,
        "vertical": vertical,
        "name": practice_name,
        "phone": phone,
        "address": address,
        "services": services,
        "hours": hours,
        "insurance": insurance or ([] if not template["insurance_supported"] else []),
        "booking_rules": {
            "min_notice_hours": 24,
            "max_advance_days": 60,
            "emergency_same_day": template["emergency_same_day"],
            "slot_duration_minutes": {s: 30 for s in services},  # Default 30 min
        },
        "tone": custom_tone or template["tone"],
        "greeting": custom_greeting or template["default_greeting"].replace(
            "our studio", practice_name
        ).replace("our office", practice_name),
        "never_say": never_say or template["never_say"],
        "off_limits_topics": template["off_limits_topics"],
        "pinecone_namespace": practice_id,
        "escalation": {
            "confidence_threshold": 0.7,
            "escalation_message": "Let me connect you with our team. Someone will reach out shortly!",
            "notify_email": f"front-desk@{practice_id}.com",
        },
    }

    # Add consultation requirements if applicable
    if template.get("requires_consultation_for"):
        config["booking_rules"]["consultation_required_for"] = [
            s for s in template["requires_consultation_for"] if s in services
        ]

    logger.info(f"Generated config for '{practice_name}' ({vertical}) with {len(services)} services")
    return config
