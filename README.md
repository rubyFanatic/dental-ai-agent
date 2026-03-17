# Dental AI Front Desk Agent

A multi-turn conversational AI agent for dental practices built with a production-grade tech stack.
Handles appointment booking, service inquiries, and patient follow-up via SMS and web chat.

## Architecture

```
Patient (SMS/Web Chat/Voice)
        │
        ▼
   FastAPI Server ──────────────► Twilio (SMS) / WebSocket (Chat)
        │
        ▼
   LangGraph Agent (Multi-Turn Orchestration)
        │
        ├── Intent Classifier Node ──► OpenAI GPT-4o
        ├── Service Lookup Node ────► Pinecone (RAG)
        ├── Availability Node ──────► Calendar API (Function Calling)
        ├── Booking Node ───────────► Booking API (Function Calling)
        ├── Guardrail Node ─────────► Service Verification
        └── Escalation Node ────────► Human Handoff
        │
        ▼
   LangSmith (Tracing + Evaluation)
```

## Tech Stack

| Layer              | Technology                        |
|--------------------|-----------------------------------|
| Language           | Python 3.11+                      |
| Orchestration      | LangGraph                         |
| LLM                | OpenAI GPT-4o (Function Calling)  |
| Knowledge Base     | Pinecone (per-practice namespace) |
| Observability      | LangSmith                         |
| API Server         | FastAPI                           |
| SMS Channel        | Twilio                            |
| Database           | PostgreSQL                        |
| Conversation State | LangGraph Checkpointer            |

## Quick Start

```bash
# 1. Clone and install
cd dental-ai-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Seed the knowledge base
python scripts/seed_knowledge.py

# 4. Test in CLI mode (no Twilio needed)
python scripts/test_conversation.py

# 5. Run the server
uvicorn app.server:app --reload --port 8000
```

## Project Structure

```
dental-ai-agent/
├── app/
│   ├── __init__.py
│   ├── server.py              # FastAPI entry point + feedback endpoint
│   ├── agent/
│   │   ├── state.py           # LangGraph state definition
│   │   ├── graph.py           # Graph construction + compilation
│   │   ├── nodes.py           # All node implementations
│   │   └── prompts.py         # System prompts per vertical
│   ├── tools/
│   │   ├── definitions.py     # OpenAI function calling schemas
│   │   └── calendar.py        # Calendar + booking tool implementations
│   ├── knowledge/
│   │   ├── setup.py           # Pinecone index creation
│   │   └── retriever.py       # RAG retrieval logic
│   ├── channels/
│   │   └── sms.py             # Twilio SMS adapter
│   ├── config/
│   │   ├── settings.py        # Environment + app settings
│   │   └── practice_config.py # Practice profiles (dental, medspa)
│   └── eval/                  # ← PROJECT 2 + 3
│       ├── evaluator.py       # Baseline scenarios + F1 scoring
│       ├── judge.py           # LLM-as-judge + pairwise evaluator
│       ├── dataset.py         # LangSmith dataset manager
│       └── metrics.py         # Observability metrics collector
├── data/
│   └── apex_dental_services.json
├── scripts/
│   ├── seed_knowledge.py      # Seed Pinecone with practice data
│   ├── test_conversation.py   # CLI conversation tester
│   ├── run_eval.py            # ← PROJECT 2: Full eval pipeline
│   ├── run_pairwise.py        # ← PROJECT 2: A/B pairwise comparison
│   ├── manage_dataset.py      # ← PROJECT 2: LangSmith dataset CRUD
│   └── metrics_report.py      # ← PROJECT 3: Observability report
├── tests/
│   └── test_agent.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 1: Multi-Turn Agent (Core)

```bash
# Interactive CLI tester
python scripts/test_conversation.py

# Medspa vertical (same engine, different config)
python scripts/test_conversation.py --practice cary-medspa-01

# Run the API server
uvicorn app.server:app --reload --port 8000

# Test via API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I need a cleaning next Tuesday", "practice_id": "apex-dental-01"}'
```

---

## 2: Agent Evaluation Pipeline

Production eval workflow: curate dataset → run agent → LLM-as-judge → F1 metrics → ship/no-ship.

### Step 1: Create baseline dataset in LangSmith
```bash
# Requires LANGCHAIN_API_KEY in .env
python scripts/manage_dataset.py create

# View stats
python scripts/manage_dataset.py stats
```

### Step 2: Run full evaluation pipeline
```bash
# Run all 24 baseline scenarios with LLM-as-judge scoring
python scripts/run_eval.py

# Filter by category
python scripts/run_eval.py --category guardrail
python scripts/run_eval.py --category booking

# Save results to data/ directory
python scripts/run_eval.py --save

# Create dataset + run eval in one command
python scripts/run_eval.py --create-dataset --save
```

The eval pipeline produces:
- **Intent classification accuracy + F1 score** (per-intent breakdown)
- **LLM-as-judge scores** (accuracy, safety, helpfulness — each 1-5)
- **Pass rate** (% of responses safe to ship to real patients)
- **Hallucination count** (should always be 0)
- **Ship/No-ship decision** (pass rate ≥90% AND 0 hallucinations)

### Step 3: Run pairwise A/B comparison
```bash
# Compare two agent versions head-to-head
python scripts/run_pairwise.py

# Save comparison results
python scripts/run_pairwise.py --save
```

### Step 4: Add production corrections (continuous feedback loop)
```bash
# Via CLI
python scripts/manage_dataset.py add-correction \
  --input "Do you do veneers?" \
  --response "Yes, we offer veneers!" \
  --correction "We don't currently offer veneers, but let me check with our team." \
  --reason "Hallucinated service not on practice list"

# Via API (for integrating into staff tools)
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "original_input": "Do you do veneers?",
    "agent_response": "Yes, we offer veneers!",
    "corrected_response": "We don'\''t offer veneers, but let me check with our team.",
    "correction_reason": "Hallucinated service"
  }'
```

---

## Project 3: LLM Observability Dashboard

Every conversation automatically traces to LangSmith (when configured).
This module pulls those traces and computes operational metrics.

### Generate metrics report
```bash
# Last 7 days (default)
python scripts/metrics_report.py

# Last 24 hours
python scripts/metrics_report.py --days 1

# Filter by practice + export
python scripts/metrics_report.py --practice apex-dental-01 --export
```

### Metrics tracked:
- **LLM calls per conversation** (production agents typically make 20-30)
- **End-to-end latency** (p50, p95, p99)
- **Cost per conversation** (input + output tokens × pricing)
- **Escalation rate** (target: <15%)
- **Booking completion rate**
- **Guardrail violation rate** (target: 0%)
- **Intent distribution**
- **Confidence score distribution**
- **Comparison to industry benchmarks**

### The interview line this enables:
> "Last week, my agent averaged 8 LLM calls per conversation,
> 4.2 seconds end-to-end latency, $0.03 per conversation,
> 12% escalation rate, and 0% hallucination rate."
```
