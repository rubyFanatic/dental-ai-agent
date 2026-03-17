"""
Tool Implementations — the actual functions called when the LLM invokes a tool.

For the MVP, these use mock data. In production, these would call real APIs:
- Calendar: Google Calendar API, Calendly, or practice management system
- Booking: Practice management system (Dentrix, Open Dental, etc.)
- Insurance: Eligibility verification API

The important thing for the interview: the ARCHITECTURE is real.
Swapping mock → real API is a configuration change, not a redesign.
"""
import json
import uuid
from datetime import datetime, timedelta
from app.config.practice_config import get_practice


# ── In-memory storage for MVP ─────────────────────────────────────────────
_bookings: dict[str, dict] = {}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Route a function call to the correct implementation.
    Returns a JSON string that gets fed back into the LLM as a tool result.
    """
    handlers = {
        "check_availability": check_availability,
        "book_appointment": book_appointment,
        "lookup_service_info": lookup_service_info,
        "check_insurance": check_insurance,
        "escalate_to_human": escalate_to_human,
    }

    handler = handlers.get(tool_name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = handler(**arguments)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Tool: Check Availability ──────────────────────────────────────────────

def check_availability(service_name: str, preferred_date: str, practice_id: str) -> dict:
    """
    Check available appointment slots.
    MVP: Returns mock availability. Production: calls calendar API.
    """
    practice = get_practice(practice_id)
    duration = practice["booking_rules"]["slot_duration_minutes"].get(service_name)

    if not duration:
        return {
            "available": False,
            "message": f"Service '{service_name}' is not offered at {practice['name']}.",
            "slots": [],
        }

    # Parse the preferred date
    try:
        date = datetime.strptime(preferred_date, "%Y-%m-%d")
    except ValueError:
        return {
            "available": False,
            "message": f"Could not parse date: {preferred_date}. Please use YYYY-MM-DD format.",
            "slots": [],
        }

    # Check if the practice is open that day
    day_name = date.strftime("%A").lower()
    hours = practice["hours"].get(day_name, "Closed")

    if hours == "Closed":
        return {
            "available": False,
            "message": f"{practice['name']} is closed on {date.strftime('%A')}s.",
            "slots": [],
            "suggestion": f"We're open {_get_next_open_day(practice, date)}. Would that work?",
        }

    # Generate mock available slots (in production, query actual calendar)
    slots = _generate_mock_slots(date, hours, duration)

    return {
        "available": len(slots) > 0,
        "date": preferred_date,
        "service": service_name,
        "duration_minutes": duration,
        "slots": slots,
        "message": f"I found {len(slots)} available times for {service_name} on {date.strftime('%A, %B %d')}." if slots else "No availability on that date.",
    }


def _generate_mock_slots(date: datetime, hours_str: str, duration: int) -> list[dict]:
    """Generate realistic-looking available time slots."""
    try:
        open_str, close_str = hours_str.split(" - ")
        open_time = datetime.strptime(open_str.strip(), "%I:%M %p").replace(
            year=date.year, month=date.month, day=date.day
        )
        close_time = datetime.strptime(close_str.strip(), "%I:%M %p").replace(
            year=date.year, month=date.month, day=date.day
        )
    except ValueError:
        return []

    slots = []
    current = open_time
    while current + timedelta(minutes=duration) <= close_time:
        # Simulate some slots being taken (every other slot roughly)
        if current.hour not in [12, 13] and current.minute == 0:  # Skip lunch, only top-of-hour
            slots.append({
                "start_time": current.strftime("%I:%M %p"),
                "end_time": (current + timedelta(minutes=duration)).strftime("%I:%M %p"),
                "datetime_iso": current.isoformat(),
            })
        current += timedelta(minutes=duration)

    # Return at most 4 slots to keep it manageable
    return slots[:4]


def _get_next_open_day(practice: dict, from_date: datetime) -> str:
    """Find the next day the practice is open."""
    for i in range(1, 8):
        next_date = from_date + timedelta(days=i)
        day_name = next_date.strftime("%A").lower()
        if practice["hours"].get(day_name, "Closed") != "Closed":
            return next_date.strftime("%A, %B %d")
    return "next week"


# ── Tool: Book Appointment ────────────────────────────────────────────────

def book_appointment(
    patient_name: str,
    service_name: str,
    appointment_datetime: str,
    practice_id: str,
    patient_phone: str = "",
    notes: str = "",
) -> dict:
    """
    Confirm and create an appointment.
    MVP: Stores in memory. Production: calls booking API.
    """
    booking_id = f"APT-{uuid.uuid4().hex[:8].upper()}"

    booking = {
        "booking_id": booking_id,
        "patient_name": patient_name,
        "patient_phone": patient_phone,
        "service": service_name,
        "datetime": appointment_datetime,
        "practice_id": practice_id,
        "notes": notes,
        "status": "confirmed",
        "created_at": datetime.now().isoformat(),
    }

    _bookings[booking_id] = booking

    # Parse datetime for friendly display
    try:
        dt = datetime.fromisoformat(appointment_datetime)
        friendly_time = dt.strftime("%A, %B %d at %I:%M %p")
    except ValueError:
        friendly_time = appointment_datetime

    return {
        "success": True,
        "booking_id": booking_id,
        "message": f"Appointment confirmed! {service_name} for {patient_name} on {friendly_time}.",
        "details": booking,
    }


# ── Tool: Lookup Service Info ─────────────────────────────────────────────

def lookup_service_info(service_name: str, practice_id: str) -> dict:
    """
    Look up detailed service information.
    MVP: Returns from config. Production: queries Pinecone knowledge base.
    """
    practice = get_practice(practice_id)

    # Check if service exists at this practice
    matching = [s for s in practice["services"] if s.lower() == service_name.lower()]
    if not matching:
        # Fuzzy match attempt
        matching = [s for s in practice["services"] if service_name.lower() in s.lower()]

    if not matching:
        return {
            "found": False,
            "message": f"'{service_name}' is not a service offered at {practice['name']}.",
            "available_services": practice["services"],
        }

    service = matching[0]
    duration = practice["booking_rules"]["slot_duration_minutes"].get(service, 30)

    # Mock detailed info (in production, this comes from Pinecone RAG)
    service_details = _get_service_details(service, practice["vertical"])

    return {
        "found": True,
        "service": service,
        "duration_minutes": duration,
        "description": service_details["description"],
        "preparation": service_details["preparation"],
        "what_to_expect": service_details["what_to_expect"],
    }


def _get_service_details(service: str, vertical: str) -> dict:
    """Mock service detail lookup. In production, this is RAG from Pinecone."""
    details = {
        "General Cleaning": {
            "description": "A thorough professional cleaning to remove plaque and tartar buildup, followed by polishing.",
            "preparation": "Brush and floss normally before your appointment. No special preparation needed.",
            "what_to_expect": "The appointment takes about 60 minutes. Your hygienist will clean your teeth, check for any concerns, and you'll leave with a fresh, clean smile.",
        },
        "Dental Exam": {
            "description": "A comprehensive examination of your teeth, gums, and oral health, including a discussion of any concerns.",
            "preparation": "Bring your insurance card and a list of any medications you take.",
            "what_to_expect": "About 45 minutes. The dentist will examine your teeth and gums, take any necessary X-rays, and discuss a personalized care plan.",
        },
        "Teeth Whitening": {
            "description": "Professional in-office teeth whitening for noticeably brighter teeth in one visit.",
            "preparation": "We recommend having a cleaning before whitening for the best results.",
            "what_to_expect": "About 75 minutes. You'll see results immediately, with teeth typically 4-8 shades brighter.",
        },
        "Botox": {
            "description": "A quick, minimally invasive treatment to smooth fine lines and wrinkles for a refreshed, natural look.",
            "preparation": "Avoid blood-thinning medications and supplements for a few days before. Arrive with a clean face.",
            "what_to_expect": "About 30 minutes. You may see initial results in 3-5 days with full results in 2 weeks.",
        },
        "Hydrafacial": {
            "description": "A multi-step facial treatment that cleanses, exfoliates, extracts, and hydrates your skin.",
            "preparation": "Arrive with a clean face. Avoid retinoids 48 hours before.",
            "what_to_expect": "About 60 minutes of relaxation. Your skin will look glowing and hydrated immediately.",
        },
    }

    default = {
        "description": f"A professional {service.lower()} service.",
        "preparation": "No special preparation needed. Feel free to call us with any questions.",
        "what_to_expect": "Our team will take great care of you. Please arrive 10 minutes early.",
    }

    return details.get(service, default)


# ── Tool: Check Insurance ─────────────────────────────────────────────────

def check_insurance(insurance_name: str, practice_id: str) -> dict:
    """Check if a specific insurance plan is accepted."""
    practice = get_practice(practice_id)

    if not practice["insurance"]:
        return {
            "accepted": False,
            "message": f"{practice['name']} is a cash-pay practice and does not accept insurance. We do offer financing options.",
        }

    # Fuzzy match against accepted insurance list
    accepted = practice["insurance"]
    matching = [ins for ins in accepted if insurance_name.lower() in ins.lower()]

    if matching:
        return {
            "accepted": True,
            "insurance": matching[0],
            "message": f"Yes! We accept {matching[0]}.",
        }
    else:
        return {
            "accepted": False,
            "insurance_asked": insurance_name,
            "message": f"I'm not sure if we accept {insurance_name}. Let me have our billing team verify and get back to you.",
            "accepted_plans": accepted,
        }


# ── Tool: Escalate to Human ──────────────────────────────────────────────

def escalate_to_human(
    reason: str,
    patient_phone: str = "",
    conversation_summary: str = "",
) -> dict:
    """
    Escalate conversation to human staff.
    MVP: Logs the escalation. Production: sends to inbox / Slack / email.
    """
    escalation_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"

    return {
        "escalated": True,
        "escalation_id": escalation_id,
        "reason": reason,
        "message": "I've notified our team. Someone will reach out to you shortly!",
    }
