"""
Create and manage LangSmith evaluation datasets.

Usage:
    python scripts/manage_dataset.py create           # Create baseline dataset
    python scripts/manage_dataset.py create --overwrite  # Recreate from scratch
    python scripts/manage_dataset.py stats            # Show dataset stats
    python scripts/manage_dataset.py add-correction \\
        --input "Do you do veneers?" \\
        --response "Yes we do veneers!" \\
        --correction "We don't currently offer veneers, but let me check with our team." \\
        --reason "Hallucinated service"

This is the dataset curation step of the eval workflow.
"""
import sys
import os
import argparse
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Manage LangSmith evaluation datasets")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create
    create_parser = subparsers.add_parser("create", help="Create baseline dataset")
    create_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset")
    create_parser.add_argument("--name", default="dental-agent-baseline", help="Dataset name")

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("--name", default="dental-agent-baseline")

    # Add correction
    add_parser = subparsers.add_parser("add-correction", help="Add a human correction")
    add_parser.add_argument("--input", required=True, help="Original patient message")
    add_parser.add_argument("--response", required=True, help="Agent's incorrect response")
    add_parser.add_argument("--correction", required=True, help="What the agent SHOULD have said")
    add_parser.add_argument("--reason", required=True, help="Why this was wrong")
    add_parser.add_argument("--practice", default="apex-dental-01")
    add_parser.add_argument("--name", default="dental-agent-baseline")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not api_key:
        print("ERROR: Set LANGCHAIN_API_KEY in .env to use dataset management.")
        print("Get a key at: https://smith.langchain.com/")
        sys.exit(1)

    from app.eval.dataset import DatasetManager

    dm = DatasetManager(dataset_name=args.name)

    if args.command == "create":
        logger.info(f"Creating dataset '{args.name}'...")
        dataset_id = dm.create_baseline_dataset(overwrite=args.overwrite)
        logger.info(f"Dataset ID: {dataset_id}")

        stats = dm.get_stats()
        logger.info(f"Total examples: {stats['total_examples']}")
        logger.info(f"Categories: {json.dumps(stats['by_category'], indent=2)}")

    elif args.command == "stats":
        stats = dm.get_stats()
        if not stats.get("exists"):
            logger.info(f"Dataset '{args.name}' does not exist. Run 'create' first.")
            return

        print(f"\nDataset: {stats['name']}")
        print(f"ID: {stats['id']}")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Created: {stats['created_at']}")
        print(f"\nBy category:")
        for cat, count in stats.get("by_category", {}).items():
            print(f"  {cat:25s}  {count}")
        print(f"\nBy source:")
        for src, count in stats.get("by_source", {}).items():
            print(f"  {src:25s}  {count}")

    elif args.command == "add-correction":
        dm.add_production_correction(
            original_input=args.input,
            agent_response=args.response,
            corrected_response=args.correction,
            correction_reason=args.reason,
            practice_id=args.practice,
        )
        logger.info("Correction added to dataset.")

        stats = dm.get_stats()
        logger.info(f"Dataset now has {stats['total_examples']} examples "
                    f"({stats.get('by_source', {}).get('human_correction', 0)} from corrections)")


if __name__ == "__main__":
    main()
