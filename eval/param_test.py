# -*- coding: utf-8 -*-
"""
Parameter Sensitivity Experiment:
Runs Prompt V6 across the 15 standardized evaluation queries at temperature 0.0 and 0.7.
Saves empirical measurements to eval/param_results.csv.

Usage:
    python eval/param_test.py [--provider groq] [--delay 1.0]
"""

import sys
import json
import csv
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_eval import load_test_queries, evaluate_single, RESULTS_FIELDNAMES

EVAL_DIR = Path(__file__).parent
PARAM_RESULTS_CSV = EVAL_DIR / "param_results.csv"


def run_param_experiment(
    provider: str = "groq",
    temperatures: list[float] | None = None,
    delay: float = 1.0,
    output_path: Path = PARAM_RESULTS_CSV,
) -> list[dict]:
    """Execute V6 prompt at different temperatures across 15 benchmark queries."""
    if temperatures is None:
        temperatures = [0.0, 0.3, 0.7]

    queries = load_test_queries()
    version = "V6 — RAG + ReAct"
    all_results = []

    print("\n" + "=" * 70)
    print("ShopEase AI — Parameter Sensitivity Experiment (Temperature 0.0 vs 0.7)")
    print(f"Provider: {provider} | Target Version: {version}")
    print(f"Temperatures: {temperatures} | Total Runs: {len(temperatures) * len(queries)}")
    print("=" * 70 + "\n")

    for temp in temperatures:
        print(f"\n--- Running Temperature = {temp} ---")
        for i, q in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {q['id']} ({q['category']}): {q['query'][:45]}...", end=" ", flush=True)
            res = evaluate_single(
                query=q,
                version=version,
                provider=provider,
                temperature=temp,
            )
            res["temperature"] = temp
            all_results.append(res)

            # Save after every single row
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_results)

            status = "[OK]" if res["task_success"] else "[FAIL]"
            halluc = " [HALLUC]" if res["hallucination"] else ""
            print(f"{status} ({res['latency_ms']}ms){halluc}")

            if delay > 0:
                time.sleep(delay)

    print(f"\n✅ All results saved to {output_path}")

    # Print summary
    print_summary(all_results)
    return all_results


def print_summary(results: list[dict]):
    """Print comparative table by temperature."""
    from collections import defaultdict
    stats = defaultdict(lambda: {"total": 0, "success": 0, "tool_acc": 0, "halluc": 0, "latency": []})

    for r in results:
        t = r["temperature"]
        stats[t]["total"] += 1
        stats[t]["success"] += r["task_success"]
        stats[t]["tool_acc"] += r["tool_accuracy"]
        stats[t]["halluc"] += r["hallucination"]
        stats[t]["latency"].append(r["latency_ms"])

    print("\n" + "=" * 75)
    print(f"{'Temperature':<15} {'Success Rate':<18} {'Tool Accuracy':<18} {'Hallucination':<16} {'Avg Latency'}")
    print("=" * 75)

    for t in sorted(stats.keys()):
        s = stats[t]
        n = s["total"]
        succ_pct = (s["success"] / n) * 100 if n else 0
        tool_pct = (s["tool_acc"] / n) * 100 if n else 0
        hall_pct = (s["halluc"] / n) * 100 if n else 0
        avg_lat = sum(s["latency"]) / len(s["latency"]) if s["latency"] else 0
        print(
            f"{t:<15.1f} "
            f"{succ_pct:>5.1f}% ({s['success']}/{n})      "
            f"{tool_pct:>5.1f}% ({s['tool_acc']}/{n})      "
            f"{hall_pct:>5.1f}% ({s['halluc']}/{n})     "
            f"{avg_lat:>6.0f} ms"
        )
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run temperature parameter experiment")
    parser.add_argument("--provider", default="groq", help="LLM Provider")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls")
    args = parser.parse_args()

    run_param_experiment(provider=args.provider, delay=args.delay)
