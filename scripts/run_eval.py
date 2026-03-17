"""
Run the full evaluation pipeline against the baseline dataset.

This is the script you run before shipping any prompt/model change.
It runs every baseline scenario through the agent, judges each response,
and produces a quality report with pass rate, F1, and per-category breakdown.

Usage:
    python scripts/run_eval.py                        # Full eval
    python scripts/run_eval.py --category booking     # Only booking scenarios
    python scripts/run_eval.py --category guardrail   # Only guardrail tests
    python scripts/run_eval.py --save                 # Save results to JSON
    python scripts/run_eval.py --create-dataset       # Create/update LangSmith dataset first

Production eval workflow:
    Baseline dataset → Run agent → LLM-as-judge scoring → F1 metrics → Ship/No-ship decision
"""
import sys
import os
import json
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import get_agent
from app.eval.evaluator import get_baseline_scenarios, evaluate_intent_accuracy
from app.eval.judge import LLMJudge
from app.eval.dataset import DatasetManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(
    category: str = None,
    practice_id: str = "apex-dental-01",
    save_results: bool = False,
    create_dataset: bool = False,
):
    """Run the full evaluation pipeline."""

    # Step 0: Optionally create/update the LangSmith dataset
    if create_dataset:
        logger.info("Creating/updating baseline dataset in LangSmith...")
        dm = DatasetManager()
        dataset_id = dm.create_baseline_dataset(overwrite=False)
        stats = dm.get_stats()
        logger.info(f"Dataset: {stats['total_examples']} examples")

    # Step 1: Load scenarios
    scenarios = get_baseline_scenarios(category)
    logger.info(f"Loaded {len(scenarios)} scenarios" + (f" (category: {category})" if category else ""))

    # Step 2: Initialize agent and judge
    logger.info("Initializing agent...")
    agent = get_agent()

    logger.info("Initializing LLM judge...")
    judge = LLMJudge()

    # Step 3: Run each scenario through the agent
    results = []
    intent_results = []

    for i, scenario in enumerate(scenarios):
        scenario_id = scenario["id"]
        logger.info(f"[{i+1}/{len(scenarios)}] Running: {scenario_id}")

        # Unique thread per scenario to avoid state leakage
        thread_id = f"eval-{scenario_id}-{datetime.now().strftime('%H%M%S')}"

        try:
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=scenario["input"])],
                    "practice_id": practice_id,
                    "channel": "eval",
                },
                {"configurable": {"thread_id": thread_id}},
            )

            # Extract agent's text response
            agent_response = ""
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    agent_response = msg.content
                    break

            actual_intent = result.get("intent", "unknown")
            confidence = result.get("confidence", 0.0)

            # Collect tool calls
            tools_called = []
            for msg in result.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    tools_called.extend([tc["name"] for tc in msg.tool_calls])

            results.append({
                "scenario_id": scenario_id,
                "category": scenario["category"],
                "patient_message": scenario["input"],
                "agent_response": agent_response,
                "expected_behavior": scenario["expected_behavior"],
                "expected_intent": scenario["expected_intent"],
                "actual_intent": actual_intent,
                "confidence": confidence,
                "tools_called": tools_called,
                "expected_tools": scenario["expected_tools"],
                "escalated": result.get("needs_escalation", False),
                "booking_confirmed": result.get("booking_confirmed", False),
            })

            intent_results.append({
                "expected_intent": scenario["expected_intent"],
                "actual_intent": actual_intent,
            })

            logger.info(f"  Intent: {actual_intent} (expected: {scenario['expected_intent']}) "
                       f"| Confidence: {confidence:.2f} | Tools: {tools_called}")

        except Exception as e:
            logger.error(f"  Error on {scenario_id}: {e}")
            results.append({
                "scenario_id": scenario_id,
                "category": scenario["category"],
                "patient_message": scenario["input"],
                "agent_response": f"ERROR: {str(e)}",
                "expected_behavior": scenario["expected_behavior"],
                "expected_intent": scenario["expected_intent"],
                "actual_intent": "error",
                "confidence": 0.0,
                "tools_called": [],
                "error": str(e),
            })

    # Step 4: Intent classification accuracy (F1)
    logger.info("\n" + "=" * 60)
    logger.info("INTENT CLASSIFICATION METRICS")
    logger.info("=" * 60)

    intent_metrics = evaluate_intent_accuracy(intent_results)
    logger.info(f"  Overall accuracy: {intent_metrics['accuracy']:.1%} ({intent_metrics['correct']}/{intent_metrics['total']})")
    logger.info(f"  Macro F1: {intent_metrics['macro_f1']:.3f}")

    for intent, m in intent_metrics["per_intent"].items():
        logger.info(f"    {intent:25s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  (n={m['support']})")

    # Step 5: LLM-as-judge quality scoring
    logger.info("\n" + "=" * 60)
    logger.info("LLM-AS-JUDGE QUALITY EVALUATION")
    logger.info("=" * 60)

    judge_results = judge.evaluate_batch(results, practice_id)
    summary = judge_results["summary"]

    logger.info(f"  Evaluated:       {summary['total_evaluated']}")
    logger.info(f"  Avg Accuracy:    {summary['avg_accuracy']:.2f}/5")
    logger.info(f"  Avg Safety:      {summary['avg_safety']:.2f}/5")
    logger.info(f"  Avg Helpfulness: {summary['avg_helpfulness']:.2f}/5")
    logger.info(f"  Avg Overall:     {summary['avg_overall']:.2f}/5")
    logger.info(f"  Pass Rate:       {summary['pass_rate']:.1%}")
    logger.info(f"  Hallucinations:  {summary['hallucination_count']} ({summary['hallucination_rate']:.1%})")
    logger.info(f"  Intent Accuracy: {summary['intent_accuracy']:.1%}")

    logger.info("\n  By Category:")
    for cat, stats in judge_results.get("by_category", {}).items():
        logger.info(f"    {cat:20s}  avg={stats['avg_score']:.1f}  pass={stats['pass_rate']:.0%}  (n={stats['count']})")

    # Show failures
    failures = [j for j in judge_results["judgments"] if not j.get("overall_pass")]
    if failures:
        logger.info(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            logger.info(f"    [{f['scenario_id']}] Score: {f.get('overall_score', '?')}/5 — {f.get('feedback', 'no feedback')}")

    # Step 6: Ship/No-ship decision
    logger.info("\n" + "=" * 60)
    ship_decision = "SHIP" if summary["pass_rate"] >= 0.90 and summary["hallucination_count"] == 0 else "DO NOT SHIP"
    logger.info(f"  DECISION: {ship_decision}")

    if ship_decision == "DO NOT SHIP":
        reasons = []
        if summary["pass_rate"] < 0.90:
            reasons.append(f"Pass rate {summary['pass_rate']:.0%} < 90% threshold")
        if summary["hallucination_count"] > 0:
            reasons.append(f"{summary['hallucination_count']} hallucination(s) detected")
        logger.info(f"  Reasons: {'; '.join(reasons)}")

    logger.info("=" * 60)

    # Step 7: Save results
    if save_results:
        output = {
            "evaluated_at": datetime.now().isoformat(),
            "practice_id": practice_id,
            "category_filter": category,
            "intent_metrics": intent_metrics,
            "judge_summary": summary,
            "judge_by_category": judge_results.get("by_category", {}),
            "ship_decision": ship_decision,
            "detailed_results": results,
            "detailed_judgments": [
                {k: v for k, v in j.items() if k != "_raw_judge_response"}
                for j in judge_results["judgments"]
            ],
        }

        filename = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", filename)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {filepath}")

    return {
        "intent_metrics": intent_metrics,
        "judge_summary": summary,
        "ship_decision": ship_decision,
    }


def main():
    parser = argparse.ArgumentParser(description="Run agent evaluation pipeline")
    parser.add_argument("--category", default=None,
                       choices=["booking", "mid_flow", "question", "insurance",
                               "edge_case", "escalation", "greeting", "farewell", "guardrail"],
                       help="Filter scenarios by category")
    parser.add_argument("--practice", default="apex-dental-01", help="Practice ID")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--create-dataset", action="store_true", help="Create LangSmith dataset first")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env first")
        sys.exit(1)

    run_evaluation(
        category=args.category,
        practice_id=args.practice,
        save_results=args.save,
        create_dataset=args.create_dataset,
    )


if __name__ == "__main__":
    main()
