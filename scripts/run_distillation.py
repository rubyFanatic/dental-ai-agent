"""
Run the model distillation pipeline.

Usage:
    python scripts/run_distillation.py generate              # Generate teacher data
    python scripts/run_distillation.py finetune              # Launch fine-tuning job
    python scripts/run_distillation.py status --job-id ft-xx # Check job status
    python scripts/run_distillation.py evaluate --model ft:gpt-4o-mini-xxx  # Compare
"""
import sys
import os
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Model distillation pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate teacher training data")
    gen_parser.add_argument("--output", default="data/distillation_training.jsonl")
    gen_parser.add_argument("--practice", default="apex-dental-01")

    # Fine-tune
    ft_parser = subparsers.add_parser("finetune", help="Launch fine-tuning job")
    ft_parser.add_argument("--training-file", default="data/distillation_training.jsonl")
    ft_parser.add_argument("--suffix", default="dental-agent")

    # Status
    status_parser = subparsers.add_parser("status", help="Check fine-tuning job status")
    status_parser.add_argument("--job-id", required=True)

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Compare teacher vs fine-tuned student")
    eval_parser.add_argument("--model", required=True, help="Fine-tuned model ID (ft:gpt-4o-mini-...)")
    eval_parser.add_argument("--practice", default="apex-dental-01")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env")
        sys.exit(1)

    from app.eval.distillation import DistillationPipeline, TRAINING_CONVERSATIONS

    if args.command == "generate":
        pipeline = DistillationPipeline(practice_id=args.practice)

        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.output,
        )
        pipeline.generate_teacher_data(TRAINING_CONVERSATIONS, output_path)

        # Show stats
        with open(output_path) as f:
            lines = f.readlines()
        logger.info(f"\nGenerated {len(lines)} training examples")
        logger.info(f"File: {output_path}")
        logger.info(f"Size: {os.path.getsize(output_path):,} bytes")
        logger.info("\nNext step: python scripts/run_distillation.py finetune")

    elif args.command == "finetune":
        pipeline = DistillationPipeline()
        training_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.training_file,
        )

        if not os.path.exists(training_path):
            print(f"Training file not found: {training_path}")
            print("Run 'generate' first.")
            sys.exit(1)

        result = pipeline.launch_finetune(training_path, suffix=args.suffix)

        # Save job info
        job_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "finetune_job.json",
        )
        with open(job_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nJob ID: {result['job_id']}")
        print(f"Status: {result['status']}")
        print(f"Saved to: {job_file}")
        print(f"\nCheck status: python scripts/run_distillation.py status --job-id {result['job_id']}")

    elif args.command == "status":
        pipeline = DistillationPipeline()
        result = pipeline.check_status(args.job_id)

        print(f"\nJob: {result['job_id']}")
        print(f"Status: {result['status']}")
        if result.get("fine_tuned_model"):
            print(f"Model: {result['fine_tuned_model']}")
            print(f"\nEvaluate: python scripts/run_distillation.py evaluate --model {result['fine_tuned_model']}")
        if result.get("trained_tokens"):
            print(f"Trained tokens: {result['trained_tokens']:,}")

    elif args.command == "evaluate":
        pipeline = DistillationPipeline(practice_id=args.practice)

        test_messages = [
            "I need a teeth cleaning next Tuesday",
            "How long does a root canal take?",
            "Do you take Delta Dental?",
            "My tooth hurts really bad, can I come in today?",
            "Do you do veneers?",
            "I want to talk to a real person",
            "Can I reschedule my appointment?",
            "What should I expect during a whitening?",
            "Thanks, see you Tuesday!",
            "Necesito una limpieza dental",
        ]

        logger.info(f"Comparing {pipeline.teacher_model} vs {args.model} on {len(test_messages)} messages...")
        results = pipeline.compare_models(args.model, test_messages)

        s = results["summary"]
        print("\n" + "=" * 60)
        print("  DISTILLATION COMPARISON REPORT")
        print("=" * 60)
        print(f"  Teacher:  {s['teacher_model']}")
        print(f"  Student:  {s['student_model']}")
        print(f"  Compared: {s['total_compared']} messages")
        print()
        print(f"  LATENCY")
        print(f"    Teacher avg:    {s['avg_teacher_latency']:.3f}s")
        print(f"    Student avg:    {s['avg_student_latency']:.3f}s")
        print(f"    Reduction:      {s['latency_reduction_pct']:.1f}%")
        print()
        print(f"  COST")
        print(f"    Teacher/conv:   ${s['est_teacher_cost_per_conv']:.6f}")
        print(f"    Student/conv:   ${s['est_student_cost_per_conv']:.6f}")
        print(f"    Reduction:      {s['cost_reduction_pct']:.1f}%")
        print()
        print(f"  TOKENS")
        print(f"    Teacher avg:    {s['avg_teacher_tokens']} tokens")
        print(f"    Student avg:    {s['avg_student_tokens']} tokens")
        print("=" * 60)

        # Save
        output_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", f"distillation_results_{pipeline.teacher_model}_vs_student.json",
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {output_file}")
        logger.info("\nNext: Run pairwise eval to measure quality retention:")
        logger.info(f"  python scripts/run_pairwise.py")


if __name__ == "__main__":
    main()
