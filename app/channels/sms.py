"""
Twilio SMS Channel Adapter.

Handles inbound SMS via webhook and sends outbound responses.
Maps SMS messages to the LangGraph agent and back.

In production, a full messaging platform handles SMS, voice, and web chat through
its own infrastructure. This Twilio integration
mirrors the same pattern: receive → process → respond.
"""
import logging
from twilio.rest import Client
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


def send_sms(to_number: str, message: str) -> bool:
    """Send an SMS via Twilio."""
    settings = get_settings()

    if not settings.twilio_account_sid:
        logger.info(f"[SMS Mock] To: {to_number} | Message: {message}")
        return True

    try:
        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        client.messages.create(
            body=message,
            from_=settings.twilio_phone_number,
            to=to_number,
        )
        logger.info(f"SMS sent to {to_number}")
        return True
    except Exception as e:
        logger.error(f"Failed to send SMS to {to_number}: {e}")
        return False
