"""
Observability Metrics — Pull traces from LangSmith and compute operational metrics.

This is Project 3: the numbers you need to be able to quote.

"My agent averages 8 LLM calls per conversation, 4.2 seconds end-to-end latency,
$0.03 per conversation, 12% escalation rate, and 0% hallucination rate."

Production AI agent platforms track these metrics across thousands of agents
making 20-30 LLM calls per interaction. This module gives you the same visibility.

Metrics tracked:
- LLM calls per conversation
- End-to-end latency (p50, p95, p99)
- Cost per conversation (input + output tokens)
- Escalation rate
- Booking completion rate
- Intent distribution
- Confidence score distribution
- Guardrail violation rate
"""
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from langsmith import Client
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── OpenAI Pricing (per 1M tokens, as of early 2026) ─────────────────────
# Update these when models or pricing change
TOKEN_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
}


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation (trace)."""
    thread_id: str = ""
    run_id: str = ""
    start_time: datetime = None
    end_time: datetime = None
    latency_seconds: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    intent: str = ""
    confidence: float = 0.0
    escalated: bool = False
    booking_completed: bool = False
    guardrail_violation: bool = False
    turns: int = 0
    practice_id: str = ""
    channel: str = ""
    model: str = ""


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple conversations."""
    period_start: datetime = None
    period_end: datetime = None
    total_conversations: int = 0
    avg_llm_calls: float = 0.0
    avg_tool_calls: float = 0.0
    avg_latency_seconds: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    avg_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    avg_tokens: float = 0.0
    total_tokens: int = 0
    escalation_rate: float = 0.0
    booking_completion_rate: float = 0.0
    guardrail_violation_rate: float = 0.0
    avg_confidence: float = 0.0
    intent_distribution: dict = field(default_factory=dict)
    conversations: list = field(default_factory=list)


class MetricsCollector:
    """
    Collect and compute metrics from LangSmith traces.

    Usage:
        collector = MetricsCollector()
        metrics = collector.collect(days=7)
        collector.print_report(metrics)
    """

    def __init__(self):
        settings = get_settings()
        self.client = Client(api_key=settings.langchain_api_key)
        self.project_name = settings.langchain_project

    def collect(self, days: int = 7, practice_id: str = None) -> AggregateMetrics:
        """
        Collect metrics from the last N days of LangSmith traces.

        Args:
            days: Number of days to look back
            practice_id: Optional filter by practice

        Returns:
            AggregateMetrics with computed stats
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        logger.info(f"Collecting metrics from {start_time.date()} to {end_time.date()}")

        # Pull root-level runs (top-level agent invocations)
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=start_time,
            end_time=end_time,
            is_root=True,
            limit=1000,
        ))

        logger.info(f"Found {len(runs)} root traces")

        if not runs:
            return AggregateMetrics(
                period_start=start_time,
                period_end=end_time,
            )

        # Process each conversation
        conversations = []
        for run in runs:
            metrics = self._process_run(run)
            if practice_id and metrics.practice_id != practice_id:
                continue
            conversations.append(metrics)

        if not conversations:
            return AggregateMetrics(
                period_start=start_time,
                period_end=end_time,
            )

        return self._aggregate(conversations, start_time, end_time)

    def _process_run(self, run) -> ConversationMetrics:
        """Extract metrics from a single LangSmith run."""
        m = ConversationMetrics()
        m.run_id = str(run.id)
        m.start_time = run.start_time
        m.end_time = run.end_time

        # Latency
        if run.start_time and run.end_time:
            m.latency_seconds = (run.end_time - run.start_time).total_seconds()

        # Token usage
        usage = run.total_tokens or 0
        m.total_tokens = usage
        if hasattr(run, 'prompt_tokens') and run.prompt_tokens:
            m.input_tokens = run.prompt_tokens
        if hasattr(run, 'completion_tokens') and run.completion_tokens:
            m.output_tokens = run.completion_tokens

        # Metadata
        metadata = run.extra or {}
        run_metadata = metadata.get("metadata", {})
        m.practice_id = run_metadata.get("practice_id", "")
        m.channel = run_metadata.get("channel", "")
        m.thread_id = (run.extra or {}).get("configurable", {}).get("thread_id", "")

        # Model
        m.model = metadata.get("ls_model_name", "unknown")

        # Estimate cost
        m.estimated_cost_usd = self._estimate_cost(m.input_tokens, m.output_tokens, m.model)

        # Count child runs (LLM calls and tool calls)
        try:
            children = list(self.client.list_runs(
                project_name=self.project_name,
                trace_id=run.trace_id,
                is_root=False,
            ))
            m.llm_calls = sum(1 for c in children if c.run_type == "llm")
            m.tool_calls = sum(1 for c in children if c.run_type == "tool")

            # Sum tokens from child LLM calls for more accurate count
            if m.total_tokens == 0:
                for c in children:
                    if c.run_type == "llm":
                        c_tokens = c.total_tokens or 0
                        m.total_tokens += c_tokens
                        if hasattr(c, 'prompt_tokens') and c.prompt_tokens:
                            m.input_tokens += c.prompt_tokens
                        if hasattr(c, 'completion_tokens') and c.completion_tokens:
                            m.output_tokens += c.completion_tokens
                m.estimated_cost_usd = self._estimate_cost(m.input_tokens, m.output_tokens, m.model)

        except Exception as e:
            logger.debug(f"Could not fetch child runs for {run.id}: {e}")

        # Extract agent state from outputs
        outputs = run.outputs or {}
        m.intent = outputs.get("intent", "")
        m.confidence = outputs.get("confidence", 0.0)
        m.escalated = outputs.get("needs_escalation", False)
        m.booking_completed = outputs.get("booking_confirmed", False)
        m.guardrail_violation = outputs.get("guardrail_violation", False)

        return m

    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost in USD based on token usage and model."""
        pricing = TOKEN_PRICING.get(model, TOKEN_PRICING.get("gpt-4o", {}))
        input_cost = (input_tokens / 1_000_000) * pricing.get("input", 2.50)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output", 10.00)
        return round(input_cost + output_cost, 6)

    def _aggregate(
        self,
        conversations: list[ConversationMetrics],
        start: datetime,
        end: datetime,
    ) -> AggregateMetrics:
        """Compute aggregate metrics from a list of conversations."""
        n = len(conversations)
        agg = AggregateMetrics()
        agg.period_start = start
        agg.period_end = end
        agg.total_conversations = n
        agg.conversations = conversations

        # Averages
        agg.avg_llm_calls = round(sum(c.llm_calls for c in conversations) / n, 1)
        agg.avg_tool_calls = round(sum(c.tool_calls for c in conversations) / n, 1)
        agg.avg_tokens = round(sum(c.total_tokens for c in conversations) / n, 0)
        agg.total_tokens = sum(c.total_tokens for c in conversations)
        agg.avg_cost_usd = round(sum(c.estimated_cost_usd for c in conversations) / n, 4)
        agg.total_cost_usd = round(sum(c.estimated_cost_usd for c in conversations), 4)
        agg.avg_confidence = round(sum(c.confidence for c in conversations) / n, 2)

        # Latency percentiles
        latencies = sorted(c.latency_seconds for c in conversations if c.latency_seconds > 0)
        if latencies:
            agg.avg_latency_seconds = round(sum(latencies) / len(latencies), 2)
            agg.p50_latency = round(latencies[int(len(latencies) * 0.50)], 2)
            agg.p95_latency = round(latencies[int(min(len(latencies) * 0.95, len(latencies) - 1))], 2)
            agg.p99_latency = round(latencies[int(min(len(latencies) * 0.99, len(latencies) - 1))], 2)

        # Rates
        agg.escalation_rate = round(sum(1 for c in conversations if c.escalated) / n, 3)
        booking_intents = [c for c in conversations if c.intent == "book_appointment"]
        if booking_intents:
            agg.booking_completion_rate = round(
                sum(1 for c in booking_intents if c.booking_completed) / len(booking_intents), 3
            )
        agg.guardrail_violation_rate = round(
            sum(1 for c in conversations if c.guardrail_violation) / n, 3
        )

        # Intent distribution
        intents = {}
        for c in conversations:
            intent = c.intent or "unknown"
            intents[intent] = intents.get(intent, 0) + 1
        agg.intent_distribution = {k: round(v / n, 3) for k, v in sorted(intents.items())}

        return agg

    def print_report(self, metrics: AggregateMetrics):
        """Print a formatted metrics report to stdout."""
        print("\n" + "=" * 60)
        print("  AGENT OBSERVABILITY REPORT")
        print("=" * 60)

        if metrics.period_start and metrics.period_end:
            print(f"  Period: {metrics.period_start.strftime('%Y-%m-%d')} to {metrics.period_end.strftime('%Y-%m-%d')}")
        print(f"  Total Conversations: {metrics.total_conversations}")
        print()

        if metrics.total_conversations == 0:
            print("  No conversations found for this period.")
            print("  Run some test conversations first:")
            print("    python scripts/test_conversation.py")
            print("=" * 60)
            return

        print("  PERFORMANCE")
        print(f"    Avg LLM calls/conversation:  {metrics.avg_llm_calls}")
        print(f"    Avg tool calls/conversation:  {metrics.avg_tool_calls}")
        print(f"    Avg latency:                  {metrics.avg_latency_seconds}s")
        print(f"    P50 latency:                  {metrics.p50_latency}s")
        print(f"    P95 latency:                  {metrics.p95_latency}s")
        print(f"    P99 latency:                  {metrics.p99_latency}s")
        print()

        print("  COST")
        print(f"    Avg tokens/conversation:      {int(metrics.avg_tokens)}")
        print(f"    Total tokens:                 {metrics.total_tokens:,}")
        print(f"    Avg cost/conversation:        ${metrics.avg_cost_usd:.4f}")
        print(f"    Total cost:                   ${metrics.total_cost_usd:.4f}")
        print()

        print("  QUALITY")
        print(f"    Avg confidence:               {metrics.avg_confidence}")
        print(f"    Escalation rate:              {metrics.escalation_rate:.1%}")
        print(f"    Booking completion rate:       {metrics.booking_completion_rate:.1%}")
        print(f"    Guardrail violation rate:      {metrics.guardrail_violation_rate:.1%}")
        print()

        print("  INTENT DISTRIBUTION")
        for intent, pct in metrics.intent_distribution.items():
            bar = "\u2588" * int(pct * 30)
            print(f"    {intent:25s} {pct:5.1%}  {bar}")
        print()
        print("=" * 60)
