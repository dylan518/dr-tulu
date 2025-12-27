#!/usr/bin/env python3
"""
Standalone HealthBench evaluation script.
Handles deduplication and uses official scoring via common.aggregate_results.

Usage:
    python scripts/evaluate_healthbench.py <path/to/results.jsonl>
"""
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

import argparse
import json

from dotenv import load_dotenv
from samplers import common

from evaluation.health_bench_eval.healthbench_eval import HealthBenchEval
from evaluation.samplers.sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)

load_dotenv()


def load_and_deduplicate_jsonl(file_path: str) -> list:
    """Load a JSONL file, deduplicate by example_id, convert to evaluation format."""
    seen_ids = set()
    results = []
    
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            example_id = data.get("example_id")
            
            if example_id in seen_ids:
                continue
            seen_ids.add(example_id)
            
            # Convert to evaluation format (same as evaluate.py convert_to_evaluate_format)
            formatted = {
                "id": example_id,
                "row": data.get("original_data", {}),
                "response_text": data.get("final_response", ""),
                "actual_queried_prompt_messages": [
                    {"role": "user", "content": data.get("problem", "")}
                ],
                "full_traces": data.get("full_traces", {}),
            }
            results.append(formatted)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate HealthBench JSONL results")
    parser.add_argument("file_path", help="Path to the JSONL results file")
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the evaluation results (default: same dir as input)",
    )
    parser.add_argument(
        "--grader-model",
        default="gpt-4.1-mini",
        help="Model to use for grading (default: gpt-4.1-mini)",
    )
    args = parser.parse_args()

    if not Path(args.file_path).exists():
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)

    # Load and deduplicate
    print(f"Loading data from: {args.file_path}")
    results = load_and_deduplicate_jsonl(args.file_path)
    print(f"Loaded {len(results)} unique examples (deduplicated)")

    # Create grading sampler
    grader_sampler = ChatCompletionSampler(
        model=args.grader_model,
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=1000,
        temperature=0,
    )

    # Run evaluation
    print("Running HealthBench evaluation...")
    evaluator = HealthBenchEval(grader_model=grader_sampler)
    eval_results = evaluator.evaluate(results)

    # Aggregate results (official scoring)
    final_result = common.aggregate_results(eval_results)

    # Determine save path
    input_dir = Path(args.file_path).parent
    input_name = Path(args.file_path).stem
    if args.save_path is None:
        results_path = input_dir / f"{input_name}_eval_results.json"
    else:
        results_path = Path(args.save_path)

    # Save results
    results_data = {
        "task": "healthbench",
        "run_mode": "evaluation",
        "score": final_result.score,
        "metrics": final_result.metrics,
        "metadata": final_result.metadata,
        "num_examples": len(results),
    }
    
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("HEALTHBENCH EVALUATION RESULTS")
    print("=" * 60)
    print(f"Examples: {len(results)}")
    print(f"Overall Score: {final_result.score:.4f}")
    print(f"\nBy Axis:")
    for k, v in sorted(final_result.metrics.items()):
        if k.startswith("axis:") and ":std" not in k:
            print(f"  {k}: {v:.4f}")
    print(f"\nBy Theme:")
    for k, v in sorted(final_result.metrics.items()):
        if k.startswith("theme:") and ":std" not in k:
            print(f"  {k}: {v:.4f}")
    print(f"\nResults saved to: {results_path}")

    return final_result


if __name__ == "__main__":
    main()

