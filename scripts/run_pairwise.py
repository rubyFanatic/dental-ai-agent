"""
Pairwise A/B Comparison — Compare two agent versions head-to-head.

When you change a prompt, swap a model, or update the system,
this script runs both versions against the same scenarios and
picks a winner for each.

This is the standard approach for evaluating AI agent changes:
"GPT-5.1 won 65.4% of head-to-head evaluations"

Usage:
    python scripts/run_pairwise.py                   # Compare current vs. modified system prompt
    python scripts/run_pairwise.py --category booking # Only booking scenarios
    python scripts/run_pairwise.py --save             # Save results to JSON

How to use:
1. Run the baseline agent (Version A) — it uses your current prompts.py
2. Modify prompts.py (or change the model in .env)
3. This script runs both versions and compares them.

For a quick test, Version A uses your current agent and Version B uses
the same agent with a temperature tweak. In practice, you'd swap prompts or models.
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
from app.eval.evaluator import get_baseline_scenarios
from app.eval.judge import PairwiseEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def get_agent_response(agent, message: str, practice_id: str, thread_id: str) -> str:
    """Run a single message through the agent and extract the text response."""
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "practice_id": practice_id,
            "channel": "eval",
        },
        {"configurable": {"thread_id": thread_id}},
    )

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    return "[No response]"


def run_pairwise(
    category: str = None,
    practice_id: str = "apex-dental-01",
    save_results: bool = False,
):
    """
    Run pairwise comparison between Version A and Version B.

    Both versions use the same LangGraph agent. The difference is
    that Version B gets a slightly modified config. In practice,
    you would modify prompts.py or .env between versions.
    """
    scenarios = get_baseline_scenarios(category)
    logger.info(f"Loaded {len(scenarios)} scenarios for pairwise comparison")

    # Initialize two agent instances
    # In practice: Version A = current production, Version B = your proposed change
    logger.info("Initializing Agent A (current version)...")
    agent_a = get_agent()

    logger.info("Initializing Agent B (modified version)...")
    # For demo: same agent, different thread IDs (in practice: different prompts/models)
    agent_b = get_agent()

    logger.info("Initializing pairwise evaluator...")
    evaluator = PairwiseEvaluator()

    # Run both agents on each scenario
    comparison_data = []
    for i, scenario in enumerate(scenarios):
        sid = scenario["id"]
        logger.info(f"[{i+1}/{len(scenarios)}] {sid}: {scenario['input'][:50]}...")

        ts = datetime.now().strftime('%H%M%S')

        try:
            response_a = get_agent_response(
                agent_a, scenario["input"], practice_id, f"pairA-{sid}-{ts}"
            )
            response_b = get_agent_response(
                agent_b, scenario["input"], practice_id, f"pairB-{sid}-{ts}"
            )

            comparison_data.append({
                "scenario_id": sid,
                "category": scenario["category"],
                "patient_message": scenario["input"],
                "response_a": response_a,
                "response_b": response_b,
                "expected_behavior": scenario["expected_behavior"],
            })

            logger.info(f"  A: {response_a[:60]}...")
            logger.info(f"  B: {response_b[:60]}...")

        except Exception as e:
            logger.error(f"  Error: {e}")

    # Run pairwise evaluation
    logger.info("\nRunning pairwise comparisons with LLM judge...")
    results = evaluator.compare_batch(comparison_data, practice_id)

    # Print report
    s = results["summary"]
    print("\n" + "=" * 60)
    print("  PAIRWISE COMPARISON REPORT")
    print("=" * 60)
    print(f"  Total comparisons:  {s['total_compared']}")
    print(f"  Version A wins:     {s['a_wins']} ({s['a_win_rate']:.0%})")
    print(f"  Version B wins:     {s['b_wins']} ({s['b_win_rate']:.0%})")
    print(f"  Ties:               {s['ties']}")
    print(f"  Avg score A:        {s['avg_a_score']:.2f}/5")
    print(f"  Avg score B:        {s['avg_b_score']:.2f}/5")
    print()

    # Show per-scenario results
    print("  Detailed Results:")
    for r in results["results"]:
        winner = r.get("winner", "?")
        icon = {"A": "\u2190", "B": "\u2192", "tie": "="}.get(winner, "?")
        print(f"    [{r['scenario_id']:15s}] {icon} Winner: {winner}  "
              f"(A={r.get('a_score', '?')}/5, B={r.get('b_score', '?')}/5)  "
              f"{r.get('reason', '')[:50]}")

    print()

    # Decision
    if s["b_win_rate"] > s["a_win_rate"] + 0.05:  # B needs >5% improvement
        print(f"  RECOMMENDATION: Ship Version B (wins {s['b_win_rate']:.0%} vs {s['a_win_rate']:.0%})")
    elif s["a_win_rate"] > s["b_win_rate"] + 0.05:
        print(f"  RECOMMENDATION: Keep Version A (wins {s['a_win_rate']:.0%} vs {s['b_win_rate']:.0%})")
    else:
        print(f"  RECOMMENDATION: No significant difference. Keep Version A (lower risk).")

    print("=" * 60)

    # Save
    if save_results:
        output = {
            "evaluated_at": datetime.now().isoformat(),
            "summary": s,
            "detailed": results["results"],
        }
        filename = f"pairwise_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", filename)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run pairwise A/B comparison")
    parser.add_argument("--category", default=None, help="Filter by category")
    parser.add_argument("--practice", default="apex-dental-01")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env first")
        sys.exit(1)

    run_pairwise(args.category, args.practice, args.save)


if __name__ == "__main__":
    main()
