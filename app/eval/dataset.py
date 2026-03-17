"""
LangSmith Dataset Manager — Create, populate, and manage evaluation datasets.

Implements a production evaluation workflow:
1. Curate baseline dataset from representative conversations
2. Expand dataset from production traces (human corrections)
3. Use datasets for pairwise evaluations when changing prompts/models

Usage:
    from app.eval.dataset import DatasetManager
    dm = DatasetManager()
    dm.create_baseline_dataset()
    dm.add_production_correction(run_id, correction)
"""
import json
import logging
from datetime import datetime
from langsmith import Client
from app.config.settings import get_settings
from app.eval.evaluator import BASELINE_SCENARIOS

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manage LangSmith datasets for agent evaluation."""

    def __init__(self, dataset_name: str = "dental-agent-baseline"):
        settings = get_settings()
        self.client = Client(api_key=settings.langchain_api_key)
        self.dataset_name = dataset_name

    # ── Create Baseline Dataset ───────────────────────────────────────

    def create_baseline_dataset(self, overwrite: bool = False) -> str:
        """
        Create the baseline evaluation dataset from predefined scenarios.

        This is Step 1 of the eval workflow:
        "Baseline Dataset Curation — create an initial dataset to represent
        basic use cases and requirements for the agent."

        Returns the dataset ID.
        """
        # Check if dataset exists
        existing = list(self.client.list_datasets(dataset_name=self.dataset_name))
        if existing and not overwrite:
            dataset = existing[0]
            logger.info(f"Dataset '{self.dataset_name}' already exists with id={dataset.id}")
            return str(dataset.id)

        if existing and overwrite:
            logger.info(f"Deleting existing dataset '{self.dataset_name}'")
            self.client.delete_dataset(dataset_id=existing[0].id)

        # Create new dataset
        dataset = self.client.create_dataset(
            dataset_name=self.dataset_name,
            description=(
                f"Baseline evaluation dataset for dental/medspa AI agent. "
                f"Created {datetime.now().strftime('%Y-%m-%d')}. "
                f"{len(BASELINE_SCENARIOS)} scenarios covering booking, questions, "
                f"edge cases, escalation, guardrails."
            ),
        )
        logger.info(f"Created dataset '{self.dataset_name}' with id={dataset.id}")

        # Add all baseline scenarios as examples
        for scenario in BASELINE_SCENARIOS:
            self.client.create_example(
                dataset_id=dataset.id,
                inputs={
                    "message": scenario["input"],
                    "practice_id": "apex-dental-01",
                    "channel": "cli",
                    "scenario_id": scenario["id"],
                    "category": scenario["category"],
                },
                outputs={
                    "expected_intent": scenario["expected_intent"],
                    "expected_behavior": scenario["expected_behavior"],
                    "expected_tools": scenario["expected_tools"],
                },
                metadata={
                    "category": scenario["category"],
                    "created_by": "baseline",
                    "created_at": datetime.now().isoformat(),
                },
            )

        logger.info(f"Added {len(BASELINE_SCENARIOS)} examples to dataset")
        return str(dataset.id)

    # ── Add Production Corrections ────────────────────────────────────

    def add_production_correction(
        self,
        original_input: str,
        agent_response: str,
        corrected_response: str,
        correction_reason: str,
        practice_id: str = "apex-dental-01",
        run_id: str = None,
    ):
        """
        Add a human correction from production to the eval dataset.

        This is the continuous feedback loop:
        "Every human correction feeds back into the evaluation dataset."

        When practice staff correct the agent, call this to capture
        the correction as a new evaluation example.
        """
        # Get or create the dataset
        existing = list(self.client.list_datasets(dataset_name=self.dataset_name))
        if not existing:
            logger.warning("Dataset doesn't exist. Creating baseline first.")
            self.create_baseline_dataset()
            existing = list(self.client.list_datasets(dataset_name=self.dataset_name))

        dataset = existing[0]

        self.client.create_example(
            dataset_id=dataset.id,
            inputs={
                "message": original_input,
                "practice_id": practice_id,
                "channel": "production",
            },
            outputs={
                "expected_behavior": corrected_response,
                "correction_reason": correction_reason,
                "original_agent_response": agent_response,
            },
            metadata={
                "category": "production_correction",
                "created_by": "human_correction",
                "created_at": datetime.now().isoformat(),
                "source_run_id": run_id or "",
            },
        )
        logger.info(f"Added production correction to dataset: {correction_reason}")

    # ── Add From LangSmith Trace ──────────────────────────────────────

    def add_from_trace(self, run_id: str, expected_output: str, category: str = "trace"):
        """
        Add an example from an existing LangSmith trace.

        This lets you browse traces in the LangSmith UI, find interesting
        conversations, and add them directly to your eval dataset.
        """
        try:
            run = self.client.read_run(run_id)
        except Exception as e:
            logger.error(f"Could not read run {run_id}: {e}")
            return

        existing = list(self.client.list_datasets(dataset_name=self.dataset_name))
        if not existing:
            logger.error("Dataset doesn't exist. Run create_baseline_dataset() first.")
            return

        dataset = existing[0]

        self.client.create_example(
            dataset_id=dataset.id,
            inputs=run.inputs or {"message": "unknown"},
            outputs={"expected_behavior": expected_output},
            metadata={
                "category": category,
                "created_by": "from_trace",
                "source_run_id": run_id,
                "created_at": datetime.now().isoformat(),
            },
        )
        logger.info(f"Added trace {run_id} to dataset as '{category}'")

    # ── Dataset Stats ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        existing = list(self.client.list_datasets(dataset_name=self.dataset_name))
        if not existing:
            return {"exists": False, "name": self.dataset_name}

        dataset = existing[0]
        examples = list(self.client.list_examples(dataset_id=dataset.id))

        # Count by category
        categories = {}
        sources = {}
        for ex in examples:
            cat = (ex.metadata or {}).get("category", "unknown")
            src = (ex.metadata or {}).get("created_by", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            sources[src] = sources.get(src, 0) + 1

        return {
            "exists": True,
            "name": self.dataset_name,
            "id": str(dataset.id),
            "total_examples": len(examples),
            "by_category": categories,
            "by_source": sources,
            "created_at": str(dataset.created_at),
        }
