"""
Voice Channel Adapter — Retell AI integration (Project 5).

Connects voice calls to the same LangGraph agent brain.
Same conversation state whether the patient texts, calls, or uses web chat.

Architecture:
    Phone call → Retell AI (STT) → Webhook → LangGraph Agent → Response → Retell AI (TTS) → Patient

Retell AI handles:
    - Speech-to-text transcription
    - Text-to-speech response
    - Interruption detection
    - Silence handling
    - Turn-taking

We handle:
    - Conversation logic (same LangGraph agent)
    - Channel-specific adaptations (shorter responses for voice)
    - Shared state across channels
"""
import json
import logging
from pydantic import BaseModel
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


# ── Retell AI Configuration ───────────────────────────────────────────────

class RetellConfig(BaseModel):
    """Configuration for a Retell AI voice agent."""
    agent_name: str = "Dental Front Desk"
    voice_id: str = "11labs-Adrian"  # ElevenLabs voice
    language: str = "en-US"
    interruption_sensitivity: float = 0.8  # 0-1, higher = easier to interrupt
    ambient_sound: str = "office"
    response_delay_ms: int = 500
    end_call_after_silence_ms: int = 30000  # 30 seconds of silence
    max_call_duration_ms: int = 600000  # 10 minutes
    opt_out_sensitive_data_storage: bool = True


# ── Voice-Specific Response Adapter ───────────────────────────────────────

def adapt_response_for_voice(text_response: str) -> str:
    """
    Adapt an agent response for voice delivery.

    Voice responses should be:
    - Shorter (1-2 sentences vs 2-3 for text)
    - More conversational
    - No URLs, links, or formatting
    - No emojis
    - Use verbal cues for lists ("first... second...")
    """
    response = text_response

    # Remove emojis
    import re
    response = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F900-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]+',
        '', response
    )

    # Remove URLs
    response = re.sub(r'https?://\S+', '', response)

    # Remove markdown-style formatting
    response = response.replace('**', '').replace('*', '').replace('_', '')

    # Shorten if too long (voice should be under ~30 words per turn)
    sentences = response.split('. ')
    if len(sentences) > 3:
        response = '. '.join(sentences[:2]) + '.'

    return response.strip()


# ── Retell Webhook Handler Models ─────────────────────────────────────────

class RetellWebhookRequest(BaseModel):
    """Inbound webhook from Retell AI when patient speaks."""
    call_id: str
    agent_id: str = ""
    transcript: str = ""  # What the patient said (STT result)
    from_number: str = ""
    to_number: str = ""
    call_status: str = ""  # "in_progress", "ended"
    metadata: dict = {}


class RetellWebhookResponse(BaseModel):
    """Response sent back to Retell AI for TTS."""
    response: str  # Text to speak
    end_call: bool = False  # Whether to end the call after this response


def build_retell_response(agent_response: str, conversation_ended: bool = False) -> RetellWebhookResponse:
    """Build the webhook response for Retell AI."""
    voice_response = adapt_response_for_voice(agent_response)

    return RetellWebhookResponse(
        response=voice_response,
        end_call=conversation_ended,
    )


# ── Retell Agent Setup Helper ─────────────────────────────────────────────

def create_retell_agent_config(practice_id: str, webhook_url: str) -> dict:
    """
    Generate the configuration to create a Retell AI agent.

    Use this to set up the agent in Retell's dashboard or API:
        retell_client.create_agent(**config)
    """
    from app.config.practice_config import get_practice
    practice = get_practice(practice_id)

    config = RetellConfig(
        agent_name=f"{practice['name']} Front Desk",
    )

    return {
        "agent_name": config.agent_name,
        "voice_id": config.voice_id,
        "language": config.language,
        "webhook_url": f"{webhook_url}/webhook/voice",
        "interruption_sensitivity": config.interruption_sensitivity,
        "ambient_sound": config.ambient_sound,
        "response_delay_ms": config.response_delay_ms,
        "end_call_after_silence_ms": config.end_call_after_silence_ms,
        "max_call_duration_ms": config.max_call_duration_ms,
        "opt_out_sensitive_data_storage": config.opt_out_sensitive_data_storage,
        "metadata": {
            "practice_id": practice_id,
            "practice_name": practice["name"],
        },
        "begin_message": practice["greeting"],
    }
