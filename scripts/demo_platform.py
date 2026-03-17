"""
Platform Demo — Show how the same engine serves multiple verticals.

Runs the same conversation through dental and medspa agents side-by-side
to demonstrate that adding a new vertical is config, not code.

Usage:
    python scripts/demo_platform.py
    python scripts/demo_platform.py --validate        # Validate all practice configs
    python scripts/demo_platform.py --onboard         # Demo new practice onboarding
"""
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.platform.factory import PlatformFactory, onboard_new_practice, VERTICAL_TEMPLATES


C_GREEN = "\033[92m"
C_BLUE = "\033[94m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_GRAY = "\033[90m"
C_BOLD = "\033[1m"
C_END = "\033[0m"


def demo_verticals():
    """Show the platform serving multiple verticals."""
    factory = PlatformFactory()

    print(f"\n{C_BOLD}{'='*60}{C_END}")
    print(f"{C_GREEN}{C_BOLD}  MULTI-VERTICAL PLATFORM DEMO{C_END}")
    print(f"{C_BOLD}{'='*60}{C_END}\n")

    print(f"{C_CYAN}Registered Verticals:{C_END} {', '.join(factory.get_verticals())}\n")

    # Show each vertical's template
    for vertical in factory.get_verticals():
        template = factory.get_vertical_template(vertical)
        print(f"  {C_BOLD}{vertical.upper()}{C_END}")
        print(f"    Tone: {template['tone']}")
        print(f"    Terminology: {template['terminology']}")
        print(f"    Insurance: {'Yes' if template['insurance_supported'] else 'No (cash pay)'}")
        print(f"    Same-day emergency: {'Yes' if template['emergency_same_day'] else 'No'}")
        print()

    # Test same message across verticals
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{C_YELLOW}Set OPENAI_API_KEY to run live agent comparison.{C_END}")
        print("Showing config comparison only.\n")
        return

    from langchain_core.messages import HumanMessage, AIMessage

    print(f"\n{C_BOLD}Side-by-Side: Same question, different verticals{C_END}\n")

    test_messages = [
        "Hi, I'd like to book an appointment",
        "Do you take insurance?",
        "What services do you offer?",
    ]

    practices = [
        ("apex-dental-01", "Dental"),
        ("cary-medspa-01", "Medspa"),
    ]

    for msg in test_messages:
        print(f"  {C_BLUE}Patient:{C_END} {msg}\n")

        for practice_id, label in practices:
            agent = factory.create_agent(practice_id)
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=msg)],
                    "practice_id": practice_id,
                    "channel": "demo",
                },
                {"configurable": {"thread_id": f"demo-{practice_id}-{msg[:10]}"}},
            )

            response = ""
            for m in reversed(result.get("messages", [])):
                if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                    response = m.content
                    break

            print(f"  {C_GREEN}[{label:8s}]{C_END} {response[:120]}...")
        print(f"  {C_GRAY}{'─'*55}{C_END}\n")


def demo_validation():
    """Validate all practice configurations."""
    factory = PlatformFactory()
    from app.config.practice_config import get_all_practice_ids

    print(f"\n{C_BOLD}Practice Configuration Validation{C_END}\n")

    for pid in get_all_practice_ids():
        result = factory.validate_practice_config(pid)
        status = f"{C_GREEN}VALID{C_END}" if result["valid"] else f"{C_YELLOW}ISSUES{C_END}"
        print(f"  {status}  {pid} ({result['vertical']})")
        print(f"         Services: {result['services_count']} | Insurance: {result['has_insurance']}")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"         {C_YELLOW}Issue: {issue}{C_END}")
        if result["warnings"]:
            for warn in result["warnings"]:
                print(f"         {C_GRAY}Warning: {warn}{C_END}")
        print()


def demo_onboard():
    """Demo onboarding a new practice."""
    print(f"\n{C_BOLD}New Practice Onboarding Demo{C_END}\n")

    # Onboard a PT clinic (new vertical)
    config = onboard_new_practice(
        practice_id="raleigh-pt-01",
        practice_name="Triangle Physical Therapy",
        vertical="pt_clinic",
        services=[
            "Initial Evaluation",
            "Follow-Up Session",
            "Manual Therapy",
            "Dry Needling",
            "Sports Rehabilitation",
            "Post-Surgical Rehab",
        ],
        hours={
            "monday": "7:00 AM - 6:00 PM",
            "tuesday": "7:00 AM - 6:00 PM",
            "wednesday": "7:00 AM - 6:00 PM",
            "thursday": "7:00 AM - 6:00 PM",
            "friday": "7:00 AM - 4:00 PM",
            "saturday": "Closed",
            "sunday": "Closed",
        },
        phone="+19195557890",
        address="789 Walnut St, Raleigh, NC 27601",
        insurance=["Blue Cross Blue Shield", "Aetna", "UnitedHealthcare", "Cigna"],
    )

    print(f"  Generated config for: {config['name']}")
    print(f"  Vertical: {config['vertical']}")
    print(f"  Services: {len(config['services'])}")
    print(f"  Insurance: {len(config['insurance'])} plans accepted")
    print(f"  Tone: {config['tone']}")
    print(f"  Greeting: {config['greeting'][:60]}...")
    print(f"\n  Full config:")
    print(f"  {json.dumps(config, indent=2)[:500]}...")
    print(f"\n  {C_GREEN}Ready to deploy. No code changes needed.{C_END}")


def main():
    parser = argparse.ArgumentParser(description="Multi-vertical platform demo")
    parser.add_argument("--validate", action="store_true", help="Validate all practice configs")
    parser.add_argument("--onboard", action="store_true", help="Demo new practice onboarding")
    args = parser.parse_args()

    if args.validate:
        demo_validation()
    elif args.onboard:
        demo_onboard()
    else:
        demo_verticals()
        demo_validation()


if __name__ == "__main__":
    main()
