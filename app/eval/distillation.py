"""
Model Distillation Pipeline — Project 4.

Pattern: Use an expensive model (GPT-4o) as a "teacher" to generate
high-quality responses, then fine-tune a cheaper model (GPT-4o-mini)
on those responses to replicate quality at lower cost and latency.

Workflow:
    1. Run baseline scenarios through GPT-4o (teacher) → collect responses
    2. Format as OpenAI fine-tuning JSONL
    3. Upload and launch fine-tuning job
    4. Evaluate fine-tuned model vs teacher with pairwise comparison
    5. Measure cost and latency savings

Usage:
    python scripts/run_distillation.py generate    # Step 1-2: Generate teacher data
    python scripts/run_distillation.py finetune    # Step 3: Launch fine-tuning job
    python scripts/run_distillation.py status      # Check fine-tuning job status
    python scripts/run_distillation.py evaluate    # Step 4-5: Compare teacher vs student
"""
import json
import logging
import os
import time
from datetime import datetime
from openai import OpenAI
from app.config.settings import get_settings
from app.config.practice_config import get_practice
from app.agent.prompts import build_system_prompt

logger = logging.getLogger(__name__)


class DistillationPipeline:
    """End-to-end model distillation from teacher to student."""

    def __init__(
        self,
        teacher_model: str = "gpt-4o",
        student_base_model: str = "gpt-4o-mini-2024-07-18",
        practice_id: str = "apex-dental-01",
    ):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.teacher_model = teacher_model
        self.student_base_model = student_base_model
        self.practice_id = practice_id
        self.practice = get_practice(practice_id)
        self.system_prompt = build_system_prompt(self.practice).replace("{channel}", "sms")

    # ── Step 1: Generate Teacher Data ─────────────────────────────────

    def generate_teacher_data(self, conversations: list[dict], output_path: str) -> str:
        """
        Run conversations through the teacher model and save as fine-tuning JSONL.

        Args:
            conversations: List of dicts with "messages" (list of user/assistant turns)
            output_path: Where to save the JSONL file

        Returns:
            Path to the generated JSONL file
        """
        training_examples = []

        for i, conv in enumerate(conversations):
            logger.info(f"Generating teacher response {i+1}/{len(conversations)}")

            messages = [{"role": "system", "content": self.system_prompt}]

            # Build multi-turn conversation
            for turn in conv.get("turns", []):
                messages.append({"role": "user", "content": turn["user"]})

                # Get teacher response
                response = self.client.chat.completions.create(
                    model=self.teacher_model,
                    messages=messages,
                    temperature=0.3,
                )

                teacher_reply = response.choices[0].message.content
                messages.append({"role": "assistant", "content": teacher_reply})

                # Log token usage
                usage = response.usage
                logger.info(f"  Turn: {turn['user'][:50]}... → {teacher_reply[:50]}... "
                           f"({usage.total_tokens} tokens)")

            # Save as training example (full conversation)
            training_examples.append({"messages": messages})

        # Write JSONL
        with open(output_path, "w") as f:
            for example in training_examples:
                f.write(json.dumps(example) + "\n")

        logger.info(f"Generated {len(training_examples)} training examples → {output_path}")
        return output_path

    # ── Step 2: Launch Fine-Tuning Job ────────────────────────────────

    def launch_finetune(self, training_file_path: str, suffix: str = "dental-agent") -> dict:
        """
        Upload training data and launch an OpenAI fine-tuning job.

        Returns:
            Dict with job_id, status, and file_id
        """
        # Upload the training file
        logger.info(f"Uploading training file: {training_file_path}")
        with open(training_file_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="fine-tune")

        logger.info(f"File uploaded: {file_obj.id}")

        # Wait for file processing
        logger.info("Waiting for file processing...")
        time.sleep(5)

        # Create fine-tuning job
        logger.info(f"Launching fine-tuning job on {self.student_base_model}...")
        job = self.client.fine_tuning.jobs.create(
            training_file=file_obj.id,
            model=self.student_base_model,
            suffix=suffix,
            hyperparameters={
                "n_epochs": 3,
            },
        )

        result = {
            "job_id": job.id,
            "status": job.status,
            "model": job.model,
            "file_id": file_obj.id,
            "created_at": datetime.now().isoformat(),
        }

        logger.info(f"Fine-tuning job created: {job.id} (status: {job.status})")
        return result

    # ── Step 3: Check Job Status ──────────────────────────────────────

    def check_status(self, job_id: str) -> dict:
        """Check the status of a fine-tuning job."""
        job = self.client.fine_tuning.jobs.retrieve(job_id)

        result = {
            "job_id": job.id,
            "status": job.status,
            "model": job.model,
            "fine_tuned_model": job.fine_tuned_model,
            "trained_tokens": job.trained_tokens,
            "error": job.error,
        }

        if job.status == "succeeded":
            logger.info(f"Fine-tuning complete! Model: {job.fine_tuned_model}")
        elif job.status == "failed":
            logger.error(f"Fine-tuning failed: {job.error}")
        else:
            logger.info(f"Status: {job.status}")

            # Show recent events
            events = self.client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id, limit=5
            )
            for event in events.data:
                logger.info(f"  [{event.created_at}] {event.message}")

        return result

    # ── Step 4: Compare Teacher vs Student ────────────────────────────

    def compare_models(
        self,
        fine_tuned_model_id: str,
        test_messages: list[str],
    ) -> dict:
        """
        Compare teacher (GPT-4o) vs fine-tuned student on the same inputs.

        Returns cost, latency, and quality comparison.
        """
        results = []

        for msg in test_messages:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": msg},
            ]

            # Teacher response
            t_start = time.time()
            teacher_resp = self.client.chat.completions.create(
                model=self.teacher_model,
                messages=messages,
                temperature=0.3,
            )
            t_latency = time.time() - t_start

            # Student response
            s_start = time.time()
            student_resp = self.client.chat.completions.create(
                model=fine_tuned_model_id,
                messages=messages,
                temperature=0.3,
            )
            s_latency = time.time() - s_start

            results.append({
                "input": msg,
                "teacher_response": teacher_resp.choices[0].message.content,
                "student_response": student_resp.choices[0].message.content,
                "teacher_tokens": teacher_resp.usage.total_tokens,
                "student_tokens": student_resp.usage.total_tokens,
                "teacher_latency": round(t_latency, 3),
                "student_latency": round(s_latency, 3),
            })

        # Aggregate
        n = len(results)
        avg_t_latency = sum(r["teacher_latency"] for r in results) / n
        avg_s_latency = sum(r["student_latency"] for r in results) / n
        avg_t_tokens = sum(r["teacher_tokens"] for r in results) / n
        avg_s_tokens = sum(r["student_tokens"] for r in results) / n

        # Estimate cost (per 1M tokens)
        t_cost_per_conv = (avg_t_tokens / 1_000_000) * (2.50 + 10.00)  # GPT-4o
        s_cost_per_conv = (avg_s_tokens / 1_000_000) * (0.15 + 0.60)   # GPT-4o-mini

        cost_savings = ((t_cost_per_conv - s_cost_per_conv) / t_cost_per_conv * 100) if t_cost_per_conv > 0 else 0
        latency_savings = ((avg_t_latency - avg_s_latency) / avg_t_latency * 100) if avg_t_latency > 0 else 0

        summary = {
            "total_compared": n,
            "teacher_model": self.teacher_model,
            "student_model": fine_tuned_model_id,
            "avg_teacher_latency": round(avg_t_latency, 3),
            "avg_student_latency": round(avg_s_latency, 3),
            "latency_reduction_pct": round(latency_savings, 1),
            "avg_teacher_tokens": round(avg_t_tokens),
            "avg_student_tokens": round(avg_s_tokens),
            "est_teacher_cost_per_conv": round(t_cost_per_conv, 6),
            "est_student_cost_per_conv": round(s_cost_per_conv, 6),
            "cost_reduction_pct": round(cost_savings, 1),
        }

        return {"summary": summary, "details": results}


# ── Training Conversation Templates ───────────────────────────────────────

TRAINING_CONVERSATIONS = [
    # Booking flows
    {"id": "train-book-001", "turns": [
        {"user": "Hi, I need to schedule a teeth cleaning"},
    ]},
    {"id": "train-book-002", "turns": [
        {"user": "I'd like to book a dental exam for next week"},
    ]},
    {"id": "train-book-003", "turns": [
        {"user": "Can I get a whitening appointment?"},
    ]},
    {"id": "train-book-004", "turns": [
        {"user": "My tooth is killing me, can I come in today?"},
    ]},
    {"id": "train-book-005", "turns": [
        {"user": "I need to schedule my daughter for a pediatric exam"},
    ]},

    # Multi-turn booking
    {"id": "train-multi-001", "turns": [
        {"user": "Hi, I need a cleaning"},
        {"user": "Do you have anything next Tuesday?"},
    ]},
    {"id": "train-multi-002", "turns": [
        {"user": "I want to book a filling appointment"},
        {"user": "Actually, can we do a crown instead?"},
    ]},

    # Service questions
    {"id": "train-question-001", "turns": [
        {"user": "How long does a root canal take?"},
    ]},
    {"id": "train-question-002", "turns": [
        {"user": "What should I expect during a dental exam?"},
    ]},
    {"id": "train-question-003", "turns": [
        {"user": "Do you do veneers?"},
    ]},
    {"id": "train-question-004", "turns": [
        {"user": "How much does a cleaning cost?"},
    ]},

    # Insurance
    {"id": "train-insurance-001", "turns": [
        {"user": "Do you accept Delta Dental?"},
    ]},
    {"id": "train-insurance-002", "turns": [
        {"user": "I have Humana, is that covered?"},
    ]},
    {"id": "train-insurance-003", "turns": [
        {"user": "Do you take any insurance at all?"},
    ]},

    # Edge cases
    {"id": "train-edge-001", "turns": [
        {"user": "Do you guys do that thing with the lights on your teeth?"},
    ]},
    {"id": "train-edge-002", "turns": [
        {"user": "k thanks"},
    ]},
    {"id": "train-edge-003", "turns": [
        {"user": "Necesito una cita para limpieza dental"},
    ]},

    # Escalation
    {"id": "train-escalate-001", "turns": [
        {"user": "I want to speak to a real person please"},
    ]},
    {"id": "train-escalate-002", "turns": [
        {"user": "I need to file a complaint about my last visit"},
    ]},

    # Greetings and farewells
    {"id": "train-greeting-001", "turns": [
        {"user": "Hello!"},
    ]},
    {"id": "train-farewell-001", "turns": [
        {"user": "Thanks, that's all I needed. Bye!"},
    ]},
    {"id": "train-farewell-002", "turns": [
        {"user": "Perfect, see you Tuesday!"},
    ]},
]
