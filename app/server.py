"""
FastAPI Server — Main entry point for the dental AI agent.

Endpoints:
- POST /webhook/sms          — Twilio SMS webhook (inbound messages)
- POST /webhook/webchat       — Web chat messages
- POST /chat                  — Direct API (for testing / CLI tool)
- GET  /health                — Health check
- GET  /conversations/{id}    — Get conversation history

The server initializes the LangGraph agent on startup and routes
all inbound messages through it.
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables before anything else
load_dotenv()

from app.agent.graph import get_agent
from app.channels.sms import send_sms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Agent instance (initialized on startup) ───────────────────────────────
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent when the server starts."""
    global agent
    logger.info("Initializing LangGraph agent...")
    agent = get_agent()
    logger.info("Agent ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Dental AI Agent",
    description="Multi-turn conversational AI for dental practices",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request/Response Models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Direct chat API request."""
    message: str
    practice_id: str = "apex-dental-01"
    thread_id: str = "test-patient-001"
    channel: str = "webchat"


class ChatResponse(BaseModel):
    """Chat API response."""
    response: str
    thread_id: str
    intent: str = ""
    confidence: float = 0.0
    booking_confirmed: bool = False
    booking_id: str = ""
    escalated: bool = False


# ── Main Chat Endpoint (Direct API) ──────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the dental agent and get a response.
    This is the primary endpoint for testing and web chat integration.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Build the config for LangGraph (thread_id enables multi-turn)
    config = {
        "configurable": {"thread_id": request.thread_id},
        "metadata": {
            "practice_id": request.practice_id,
            "channel": request.channel,
        },
    }

    # Build initial state for new conversations or just pass the message
    input_state = {
        "messages": [HumanMessage(content=request.message)],
        "practice_id": request.practice_id,
        "channel": request.channel,
    }

    try:
        # Invoke the LangGraph agent
        result = agent.invoke(input_state, config)

        # Extract the agent's response (last AI message)
        response_text = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                response_text = msg.content
                break

        if not response_text:
            response_text = "I'm sorry, I couldn't process that. Could you try again?"

        return ChatResponse(
            response=response_text,
            thread_id=request.thread_id,
            intent=result.get("intent", ""),
            confidence=result.get("confidence", 0.0),
            booking_confirmed=result.get("booking_confirmed", False),
            booking_id=result.get("booking_id", ""),
            escalated=result.get("needs_escalation", False),
        )

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# ── Twilio SMS Webhook ────────────────────────────────────────────────────

@app.post("/webhook/sms")
async def handle_sms(request: Request):
    """
    Twilio SMS webhook. Receives inbound messages and responds.

    Twilio sends form data with:
    - Body: message text
    - From: sender phone number
    - To: your Twilio number
    """
    form_data = await request.form()
    message_body = form_data.get("Body", "")
    from_number = form_data.get("From", "")

    if not message_body or not from_number:
        return JSONResponse(content={"error": "Missing Body or From"}, status_code=400)

    logger.info(f"SMS from {from_number}: {message_body}")

    # Use phone number as thread_id for conversation continuity
    chat_request = ChatRequest(
        message=message_body,
        practice_id="apex-dental-01",  # In production, route by Twilio number
        thread_id=from_number,
        channel="sms",
    )

    response = await chat(chat_request)

    # Send response back via SMS
    send_sms(from_number, response.response)

    # Return TwiML empty response (we send SMS separately)
    return JSONResponse(content={"status": "ok"})


# ── Retell AI Voice Webhook (Project 5) ───────────────────────────────────

@app.post("/webhook/voice")
async def handle_voice(request: Request):
    """
    Retell AI voice webhook. Receives transcribed speech and returns text for TTS.

    Same LangGraph agent brain — different channel adapter.
    Responses are shortened for natural voice delivery.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    body = await request.json()
    transcript = body.get("transcript", "")
    call_id = body.get("call_id", "")
    from_number = body.get("from_number", call_id)

    if not transcript:
        return JSONResponse(content={"response": "", "end_call": False})

    logger.info(f"Voice from {from_number}: {transcript}")

    chat_request = ChatRequest(
        message=transcript,
        practice_id="apex-dental-01",
        thread_id=f"voice-{from_number}",
        channel="voice",
    )

    response = await chat(chat_request)

    # Adapt response for voice delivery
    from app.channels.voice import adapt_response_for_voice
    voice_text = adapt_response_for_voice(response.response)

    return JSONResponse(content={
        "response": voice_text,
        "end_call": response.escalated or bool(response.response and "bye" in response.response.lower()),
    })


# ── Platform Management Endpoints (Project 6) ────────────────────────────

@app.get("/platform/verticals")
async def list_verticals():
    """List all available verticals and registered practices."""
    from app.platform.factory import PlatformFactory
    from app.config.practice_config import get_all_practice_ids, get_practice

    factory = PlatformFactory()
    practices = []
    for pid in get_all_practice_ids():
        p = get_practice(pid)
        practices.append({
            "practice_id": pid,
            "name": p["name"],
            "vertical": p["vertical"],
            "services_count": len(p["services"]),
        })

    return {
        "verticals": factory.get_verticals(),
        "practices": practices,
    }


@app.get("/platform/validate/{practice_id}")
async def validate_practice(practice_id: str):
    """Validate a practice configuration against its vertical template."""
    from app.platform.factory import PlatformFactory
    factory = PlatformFactory()
    try:
        return factory.validate_practice_config(practice_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Guardrail Test Endpoint (Project 7) ───────────────────────────────────

class GuardrailTestRequest(BaseModel):
    """Test a response against guardrails without sending to a patient."""
    response_text: str
    practice_id: str = "apex-dental-01"
    intent: str = ""
    confidence: float = 1.0


@app.post("/guardrails/test")
async def test_guardrails(request: GuardrailTestRequest):
    """
    Test a response against the guardrail engine.
    Useful for debugging why a response was blocked or modified.
    """
    from app.guardrails import run_guardrails
    result = run_guardrails(
        response=request.response_text,
        practice_id=request.practice_id,
        intent=request.intent,
        confidence=request.confidence,
    )
    return {
        "passed": result.passed,
        "blocked": result.blocked,
        "regenerate": result.regenerate,
        "violations": result.violations,
        "modified_response": result.modified_response or None,
    }


# ── Feedback Endpoint (Continuous Learning Loop) ──────────────────────────

class FeedbackRequest(BaseModel):
    """Submit a human correction for a conversation."""
    original_input: str
    agent_response: str
    corrected_response: str
    correction_reason: str
    practice_id: str = "apex-dental-01"
    run_id: str = ""


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit a human correction to the evaluation dataset.

    This is the continuous feedback loop:
    Staff corrects the agent → correction becomes an eval example →
    next eval run catches this class of error → agent improves.
    """
    try:
        from app.eval.dataset import DatasetManager
        dm = DatasetManager()
        dm.add_production_correction(
            original_input=request.original_input,
            agent_response=request.agent_response,
            corrected_response=request.corrected_response,
            correction_reason=request.correction_reason,
            practice_id=request.practice_id,
            run_id=request.run_id,
        )
        return {"status": "ok", "message": "Correction added to evaluation dataset"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Dataset Stats Endpoint ────────────────────────────────────────────────

@app.get("/eval/dataset-stats")
async def dataset_stats():
    """Get evaluation dataset statistics."""
    try:
        from app.eval.dataset import DatasetManager
        dm = DatasetManager()
        return dm.get_stats()
    except Exception as e:
        return {"error": str(e)}


# ── Health Check ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent_ready": agent is not None,
    }
