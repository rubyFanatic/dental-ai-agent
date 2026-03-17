"""
Practice Configuration — Vertical baselines with business-level tuning.

Each practice gets:
- A vertical template (dental, medspa, etc.)
- Business-specific overrides (services, hours, tone, rules)
- Its own Pinecone namespace for knowledge retrieval

Adding a new practice = new config dict. No code changes to the agent.
"""

PRACTICES = {
    "apex-dental-01": {
        "practice_id": "apex-dental-01",
        "vertical": "dental",
        "name": "Apex Family Dental",
        "phone": "+19195551234",
        "address": "123 Main St, Apex, NC 27502",
        "website": "https://apexfamilydental.com",

        # Services offered (agent can ONLY reference these — guardrail enforced)
        "services": [
            "General Cleaning",
            "Deep Cleaning",
            "Teeth Whitening",
            "Dental Exam",
            "X-Rays",
            "Fillings",
            "Crown",
            "Root Canal",
            "Extraction",
            "Emergency Visit",
            "Invisalign Consultation",
            "Pediatric Dental Exam",
        ],

        # Operating hours
        "hours": {
            "monday": "8:00 AM - 5:00 PM",
            "tuesday": "8:00 AM - 5:00 PM",
            "wednesday": "8:00 AM - 5:00 PM",
            "thursday": "8:00 AM - 7:00 PM",
            "friday": "8:00 AM - 3:00 PM",
            "saturday": "Closed",
            "sunday": "Closed",
        },

        # Insurance accepted
        "insurance": [
            "Delta Dental",
            "Cigna",
            "MetLife",
            "Aetna",
            "Blue Cross Blue Shield",
            "United Healthcare",
            "Guardian",
        ],

        # Booking rules
        "booking_rules": {
            "min_notice_hours": 24,
            "max_advance_days": 60,
            "new_patient_requires": "Dental Exam",  # New patients must book exam first
            "emergency_same_day": True,
            "slot_duration_minutes": {
                "General Cleaning": 60,
                "Deep Cleaning": 90,
                "Teeth Whitening": 75,
                "Dental Exam": 45,
                "X-Rays": 30,
                "Fillings": 60,
                "Crown": 90,
                "Root Canal": 120,
                "Extraction": 60,
                "Emergency Visit": 30,
                "Invisalign Consultation": 45,
                "Pediatric Dental Exam": 45,
            },
        },

        # Tone and brand voice
        "tone": "friendly, professional, caring",
        "greeting": "Hi! This is Apex Family Dental. How can I help you today?",

        # Things the agent must NEVER say (guardrail)
        "never_say": [
            "I'm an AI",
            "I'm a chatbot",
            "I don't know",  # Should escalate instead
        ],

        # Things the agent should NEVER discuss (guardrail)
        "off_limits_topics": [
            "specific pricing",  # Say "pricing varies, we can discuss at your visit"
            "medical diagnosis",
            "competitor practices",
            "malpractice",
        ],

        # Pinecone namespace for this practice's knowledge base
        "pinecone_namespace": "apex-dental-01",

        # Escalation settings
        "escalation": {
            "confidence_threshold": 0.7,
            "escalation_message": "Let me connect you with our team for that. Someone will reach out shortly!",
            "notify_email": "front-desk@apexfamilydental.com",
        },
    },

    # ------------------------------------------------------------------
    # Second practice (medspa) — demonstrates multi-vertical capability
    # ------------------------------------------------------------------
    "cary-medspa-01": {
        "practice_id": "cary-medspa-01",
        "vertical": "medspa",
        "name": "Glow Aesthetics Cary",
        "phone": "+19195559876",
        "address": "456 Kildaire Farm Rd, Cary, NC 27511",
        "website": "https://glowcary.com",

        "services": [
            "Botox",
            "Dermal Fillers",
            "Chemical Peel",
            "Microneedling",
            "Hydrafacial",
            "Laser Hair Removal",
            "IPL Photofacial",
            "Body Contouring",
            "IV Therapy",
            "Complimentary Consultation",
        ],

        "hours": {
            "monday": "9:00 AM - 6:00 PM",
            "tuesday": "9:00 AM - 6:00 PM",
            "wednesday": "9:00 AM - 6:00 PM",
            "thursday": "9:00 AM - 7:00 PM",
            "friday": "9:00 AM - 6:00 PM",
            "saturday": "10:00 AM - 4:00 PM",
            "sunday": "Closed",
        },

        "insurance": [],  # Medspa = cash pay / financing only

        "booking_rules": {
            "min_notice_hours": 24,
            "max_advance_days": 90,
            "new_patient_requires": "Complimentary Consultation",
            "emergency_same_day": False,
            "consultation_required_for": ["Botox", "Dermal Fillers", "Body Contouring"],
            "slot_duration_minutes": {
                "Botox": 30,
                "Dermal Fillers": 45,
                "Chemical Peel": 60,
                "Microneedling": 60,
                "Hydrafacial": 60,
                "Laser Hair Removal": 45,
                "IPL Photofacial": 45,
                "Body Contouring": 90,
                "IV Therapy": 60,
                "Complimentary Consultation": 30,
            },
        },

        "tone": "luxury, warm, empowering, professional",
        "greeting": "Hi! Welcome to Glow Aesthetics. We'd love to help you look and feel your best. What can I help you with?",

        "never_say": [
            "I'm an AI",
            "I'm a chatbot",
            "cheap",
            "discount",
            "injectable",  # Use "treatment" instead
        ],

        "off_limits_topics": [
            "specific pricing over text",  # "We'd love to discuss pricing during your consultation"
            "medical diagnosis",
            "competitor practices",
            "guarantee results",
        ],

        "pinecone_namespace": "cary-medspa-01",

        "escalation": {
            "confidence_threshold": 0.7,
            "escalation_message": "Great question! Let me have one of our aesthetic specialists reach out to you personally.",
            "notify_email": "concierge@glowcary.com",
        },
    },
}


def get_practice(practice_id: str) -> dict:
    """Get practice config by ID. Raises KeyError if not found."""
    if practice_id not in PRACTICES:
        raise KeyError(f"Practice '{practice_id}' not found. Available: {list(PRACTICES.keys())}")
    return PRACTICES[practice_id]


def get_all_practice_ids() -> list[str]:
    """List all registered practice IDs."""
    return list(PRACTICES.keys())
