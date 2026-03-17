"""
OpenAI Function Calling Tool Definitions.

These are the tools the LLM can invoke during a conversation.
Each tool maps to a real function in tools/calendar.py.

Production AI agent platforms use function calling to interact with
booking systems, CRMs, and calendars in the same way:
- The model decides WHICH tool to call based on conversation context.
- Tool results are fed back into the conversation for the next response.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check available appointment slots for a specific service on a given date. Call this when a patient wants to book an appointment and you need to find open times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "The dental/medical service to book (e.g., 'General Cleaning', 'Dental Exam')"
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": "The preferred date in YYYY-MM-DD format. If patient says 'next Tuesday', convert to actual date."
                    },
                    "practice_id": {
                        "type": "string",
                        "description": "The practice identifier"
                    }
                },
                "required": ["service_name", "preferred_date", "practice_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a confirmed appointment. Only call this AFTER the patient has confirmed the time slot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Patient's full name"
                    },
                    "patient_phone": {
                        "type": "string",
                        "description": "Patient's phone number"
                    },
                    "service_name": {
                        "type": "string",
                        "description": "The service being booked"
                    },
                    "appointment_datetime": {
                        "type": "string",
                        "description": "Confirmed appointment date and time in ISO format (YYYY-MM-DDTHH:MM:SS)"
                    },
                    "practice_id": {
                        "type": "string",
                        "description": "The practice identifier"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Any special notes or patient concerns"
                    }
                },
                "required": ["patient_name", "service_name", "appointment_datetime", "practice_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_service_info",
            "description": "Look up detailed information about a specific service — description, duration, preparation instructions, and what to expect. Call this when a patient asks questions about a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "The service to look up"
                    },
                    "practice_id": {
                        "type": "string",
                        "description": "The practice identifier"
                    }
                },
                "required": ["service_name", "practice_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_insurance",
            "description": "Check if a specific insurance plan is accepted by the practice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "insurance_name": {
                        "type": "string",
                        "description": "The insurance plan or provider name"
                    },
                    "practice_id": {
                        "type": "string",
                        "description": "The practice identifier"
                    }
                },
                "required": ["insurance_name", "practice_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Transfer the conversation to a human staff member. Call this when: the patient explicitly requests a human, the question is too complex, you're not confident in the answer, or the topic is sensitive/medical.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the conversation is being escalated"
                    },
                    "patient_phone": {
                        "type": "string",
                        "description": "Patient's phone number for callback"
                    },
                    "conversation_summary": {
                        "type": "string",
                        "description": "Brief summary of the conversation so far to give the human context"
                    }
                },
                "required": ["reason"]
            }
        }
    },
]
