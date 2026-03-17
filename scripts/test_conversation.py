"""
Interactive CLI Conversation Tester.

Test the dental agent without needing Twilio, a web server, or any external setup.
Just run this script and start chatting.

Usage:
    python scripts/test_conversation.py
    python scripts/test_conversation.py --practice cary-medspa-01
    python scripts/test_conversation.py --thread my-test-001

Features:
- Multi-turn conversation with state persistence
- Shows intent classification and confidence
- Shows which tools were called
- Color-coded output
- Type 'reset' to start a new conversation
- Type 'state' to see current conversation state
- Type 'quit' to exit
"""
import sys
import os
import uuid
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.agent.graph import get_agent
from app.config.practice_config import get_practice

# Suppress noisy logs during interactive testing
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Colors for terminal output ─────────────────────────────────────────────
class C:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_banner(practice: dict):
    print(f"\n{C.BOLD}{'='*60}{C.END}")
    print(f"{C.GREEN}{C.BOLD}  Dental AI Agent — Interactive Tester{C.END}")
    print(f"{C.GRAY}  Practice: {practice['name']}{C.END}")
    print(f"{C.GRAY}  Vertical: {practice['vertical']}{C.END}")
    print(f"{C.GRAY}  Services: {len(practice['services'])} available{C.END}")
    print(f"{C.BOLD}{'='*60}{C.END}")
    print(f"{C.GRAY}  Commands:{C.END}")
    print(f"{C.GRAY}    reset  — Start a new conversation{C.END}")
    print(f"{C.GRAY}    state  — Show current agent state{C.END}")
    print(f"{C.GRAY}    tools  — Show available tools{C.END}")
    print(f"{C.GRAY}    quit   — Exit{C.END}")
    print(f"{C.BOLD}{'='*60}{C.END}\n")


def run_interactive(practice_id: str, initial_thread_id: str = None):
    """Run an interactive conversation loop."""
    practice = get_practice(practice_id)
    print_banner(practice)

    # Initialize agent
    print(f"{C.GRAY}Initializing agent...{C.END}")
    agent = get_agent()
    print(f"{C.GREEN}Agent ready!{C.END}\n")

    thread_id = initial_thread_id or f"test-{uuid.uuid4().hex[:8]}"
    turn_count = 0

    while True:
        # Get user input
        try:
            user_input = input(f"{C.BLUE}{C.BOLD}Patient:{C.END} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C.GRAY}Goodbye!{C.END}")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "quit":
            print(f"{C.GRAY}Goodbye!{C.END}")
            break

        if user_input.lower() == "reset":
            thread_id = f"test-{uuid.uuid4().hex[:8]}"
            turn_count = 0
            print(f"\n{C.YELLOW}--- New conversation started (thread: {thread_id}) ---{C.END}\n")
            continue

        if user_input.lower() == "state":
            _print_state(agent, thread_id)
            continue

        if user_input.lower() == "tools":
            from app.tools.definitions import TOOLS
            for tool in TOOLS:
                name = tool["function"]["name"]
                desc = tool["function"]["description"][:80]
                print(f"  {C.CYAN}{name}{C.END}: {C.GRAY}{desc}...{C.END}")
            print()
            continue

        # Send message to agent
        turn_count += 1
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "practice_id": practice_id,
                "channel": "cli",
                "turn": turn_count,
            },
        }

        input_state = {
            "messages": [HumanMessage(content=user_input)],
            "practice_id": practice_id,
            "channel": "cli",
        }

        try:
            result = agent.invoke(input_state, config)

            # Show debug info
            intent = result.get("intent", "?")
            confidence = result.get("confidence", 0)
            print(f"{C.GRAY}  [intent: {intent} | confidence: {confidence:.2f} | turn: {turn_count}]{C.END}")

            # Show tool calls if any
            for msg in result.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"{C.CYAN}  [tool: {tc['name']}({_truncate_args(tc['args'])})]{C.END}")
                if isinstance(msg, ToolMessage):
                    print(f"{C.CYAN}  [result: {msg.content[:100]}...]{C.END}")

            # Show the agent's response
            response_text = ""
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    response_text = msg.content
                    break

            if response_text:
                print(f"{C.GREEN}{C.BOLD}Agent:{C.END} {response_text}\n")
            else:
                print(f"{C.RED}[No response generated]{C.END}\n")

            # Show escalation / booking flags
            if result.get("needs_escalation"):
                print(f"{C.YELLOW}  ⚠ ESCALATED: {result.get('escalation_reason', 'unknown')}{C.END}\n")
            if result.get("booking_confirmed"):
                print(f"{C.GREEN}  ✓ BOOKED: {result.get('booking_id', '')}{C.END}\n")
            if result.get("conversation_ended"):
                print(f"{C.GRAY}  --- Conversation ended ---{C.END}\n")

        except Exception as e:
            print(f"{C.RED}Error: {e}{C.END}\n")
            logger.error(f"Agent error: {e}", exc_info=True)


def _truncate_args(args: dict) -> str:
    """Truncate tool arguments for display."""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 30:
            v_str = v_str[:27] + "..."
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)


def _print_state(agent, thread_id: str):
    """Print the current conversation state."""
    try:
        state = agent.get_state({"configurable": {"thread_id": thread_id}})
        values = state.values
        print(f"\n{C.YELLOW}--- Current State ---{C.END}")
        for key, value in values.items():
            if key == "messages":
                print(f"  {C.CYAN}messages:{C.END} {len(value)} messages")
                for msg in value[-4:]:  # Show last 4 messages
                    role = type(msg).__name__.replace("Message", "")
                    content = (msg.content or "[tool call]")[:60]
                    print(f"    {C.GRAY}{role}: {content}{C.END}")
            elif value:  # Only show non-empty fields
                print(f"  {C.CYAN}{key}:{C.END} {value}")
        print()
    except Exception as e:
        print(f"{C.GRAY}No state yet (start a conversation first){C.END}\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive dental agent tester")
    parser.add_argument(
        "--practice",
        default="apex-dental-01",
        choices=["apex-dental-01", "cary-medspa-01"],
        help="Practice to test (default: apex-dental-01)",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Thread ID for conversation continuity (auto-generated if not set)",
    )
    args = parser.parse_args()

    run_interactive(args.practice, args.thread)


if __name__ == "__main__":
    main()
