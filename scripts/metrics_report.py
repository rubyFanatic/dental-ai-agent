"""
Metrics Report — Pull LangSmith traces and generate an observability report.

This is Project 3. The script you run to get the numbers for the interview:

"Last week, my agent averaged 8 LLM calls per conversation,
 4.2 seconds end-to-end latency, $0.03 per conversation,
 12% escalation rate, and 0% hallucination rate."

Usage:
    python scripts/metrics_report.py                  # Last 7 days
    python scripts/metrics_report.py --days 1         # Last 24 hours
    python scripts/metrics_report.py --days 30        # Last month
    python scripts/metrics_report.py --practice apex-dental-01  # Filter by practice
    python scripts/metrics_report.py --export         # Export to JSON

Requires LANGCHAIN_API_KEY in .env for LangSmith access.
If not configured, shows instructions for running conversations first.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def run_report(days: int = 7, practice_id: str = None, export: bool = False):
    """Pull metrics from LangSmith and print/export a report."""

    api_key = os.getenv("LANGCHAIN_API_KEY", "")

    if not api_key:
        print("\n" + "=" * 60)
        print("  LANGSMITH NOT CONFIGURED")
        print("=" * 60)
        print()
        print("  To use observability metrics, add to your .env:")
        print("    LANGCHAIN_TRACING_V2=true")
        print("    LANGCHAIN_API_KEY=lsv2_...")
        print("    LANGCHAIN_PROJECT=dental-agent-dev")
        print()
        print("  Then run some conversations:")
        print("    python scripts/test_conversation.py")
        print()
        print("  Every conversation will automatically trace to LangSmith.")
        print("  Then run this script again to see your metrics.")
        print("=" * 60)
        return

    from app.eval.metrics import MetricsCollector

    collector = MetricsCollector()

    print(f"\nCollecting metrics from last {days} days...")
    metrics = collector.collect(days=days, practice_id=practice_id)
    collector.print_report(metrics)

    # Additional detail if conversations exist
    if metrics.total_conversations > 0:
        # Cost projection
        monthly_estimate = metrics.avg_cost_usd * 30 * (metrics.total_conversations / max(days, 1))
        print(f"\n  COST PROJECTION (at current volume)")
        print(f"    Daily avg conversations:   {metrics.total_conversations / max(days, 1):.0f}")
        print(f"    Estimated monthly cost:     ${monthly_estimate:.2f}")
        print(f"    Cost per 1000 conversations: ${metrics.avg_cost_usd * 1000:.2f}")
        print()

        # Comparison to industry benchmarks
        print("  COMPARISON TO INDUSTRY BENCHMARKS")
        print(f"    Your LLM calls/conv:   {metrics.avg_llm_calls:.0f}   (production target: 20-30)")
        print(f"    Your escalation rate:   {metrics.escalation_rate:.0%}   (target: <15%)")
        print(f"    Your guardrail viols:   {metrics.guardrail_violation_rate:.0%}   (target: 0%)")
        print()

        # Per-conversation breakdown for recent conversations
        print("  RECENT CONVERSATIONS (last 5)")
        recent = sorted(
            [c for c in metrics.conversations if c.start_time],
            key=lambda c: c.start_time,
            reverse=True,
        )[:5]

        for c in recent:
            time_str = c.start_time.strftime("%m/%d %H:%M") if c.start_time else "unknown"
            esc = " ESC" if c.escalated else ""
            book = " BOOKED" if c.booking_completed else ""
            guard = " GUARD!" if c.guardrail_violation else ""
            print(f"    {time_str} | {c.intent:20s} | {c.llm_calls:2d} LLM calls | "
                  f"{c.latency_seconds:5.1f}s | ${c.estimated_cost_usd:.4f} | "
                  f"conf={c.confidence:.2f}{esc}{book}{guard}")

        print()
        print("=" * 60)

    # Export
    if export and metrics.total_conversations > 0:
        output = {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "practice_filter": practice_id,
            "summary": {
                "total_conversations": metrics.total_conversations,
                "avg_llm_calls": metrics.avg_llm_calls,
                "avg_tool_calls": metrics.avg_tool_calls,
                "avg_latency_seconds": metrics.avg_latency_seconds,
                "p50_latency": metrics.p50_latency,
                "p95_latency": metrics.p95_latency,
                "p99_latency": metrics.p99_latency,
                "avg_cost_usd": metrics.avg_cost_usd,
                "total_cost_usd": metrics.total_cost_usd,
                "avg_tokens": metrics.avg_tokens,
                "total_tokens": metrics.total_tokens,
                "escalation_rate": metrics.escalation_rate,
                "booking_completion_rate": metrics.booking_completion_rate,
                "guardrail_violation_rate": metrics.guardrail_violation_rate,
                "avg_confidence": metrics.avg_confidence,
                "intent_distribution": metrics.intent_distribution,
            },
            "conversations": [
                {
                    "run_id": c.run_id,
                    "thread_id": c.thread_id,
                    "intent": c.intent,
                    "confidence": c.confidence,
                    "llm_calls": c.llm_calls,
                    "tool_calls": c.tool_calls,
                    "latency_seconds": c.latency_seconds,
                    "total_tokens": c.total_tokens,
                    "estimated_cost_usd": c.estimated_cost_usd,
                    "escalated": c.escalated,
                    "booking_completed": c.booking_completed,
                    "guardrail_violation": c.guardrail_violation,
                    "start_time": c.start_time.isoformat() if c.start_time else None,
                }
                for c in metrics.conversations
            ],
        }

        filename = f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", filename)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Report exported to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate observability metrics report")
    parser.add_argument("--days", type=int, default=7, help="Look back period in days (default: 7)")
    parser.add_argument("--practice", default=None, help="Filter by practice ID")
    parser.add_argument("--export", action="store_true", help="Export report to JSON")
    args = parser.parse_args()

    run_report(days=args.days, practice_id=args.practice, export=args.export)


if __name__ == "__main__":
    main()
