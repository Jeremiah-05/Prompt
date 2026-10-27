# -*- coding: utf-8 -*-
"""
Evaluation Runner - runs test queries against prompt versions.
Generates and updates eval/results.csv with metrics per query per version.

Now integrates eval/scoring.py for ground-truth evaluation:
  - Unified task_success rule across V1-V6
  - Hallucination flagged only on contradiction (not "by definition")
  - Wilson 95% confidence intervals in summary

Usage:
    python eval/run_eval.py [--provider groq] [--versions V5,V6]
    python eval/run_eval.py --heldout               # run held-out set on V6 only
    python eval/run_eval.py --runs 3 --versions V6   # 3 runs for spread analysis
"""

import sys
import json
import csv
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_client import LLMUnavailableError
from prompts.builder import list_versions, RECOMMENDED_TEMPS
from eval.scoring import (
    score_answer, summarize, wilson_interval, load_policy_facts,
    GROUND_TRUTH_CASES,
)

EVAL_DIR = Path(__file__).parent
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"
CASES_JSON_PATH = EVAL_DIR / "cases.json"
HELDOUT_QUERIES_PATH = EVAL_DIR / "test_queries_heldout.json"
RESULTS_CSV_PATH = EVAL_DIR / "results.csv"
RESULTS_JSON_PATH = EVAL_DIR / "results.json"
HELDOUT_RESULTS_CSV = EVAL_DIR / "heldout_results.csv"
HELDOUT_RESULTS_JSON = EVAL_DIR / "heldout_results.json"
HELDOUT_V2_QUERIES_PATH = EVAL_DIR / "test_queries_heldout_v2.json"
HELDOUT_V2_RESULTS_CSV = EVAL_DIR / "heldout_v2_results.csv"
HELDOUT_V2_RESULTS_JSON = EVAL_DIR / "heldout_v2_results.json"

# Load policy facts once at module level
try:
    POLICY_FACTS = load_policy_facts()
except FileNotFoundError:
    POLICY_FACTS = set()


def load_ground_truth_cases() -> dict:
    """Load ground-truth case specifications from cases.json or fall back to scoring.py."""
    if CASES_JSON_PATH.exists():
        try:
            with open(CASES_JSON_PATH, "r", encoding="utf-8") as f:
                cases = json.load(f)
                return {c["id"]: c for c in cases}
        except Exception:
            pass
    return GROUND_TRUTH_CASES


def load_test_queries(heldout: bool = False, queries_path: Path | None = None) -> list[dict]:
    """Load test queries from JSON file."""
    if queries_path is not None:
        path = Path(queries_path)
    else:
        path = HELDOUT_QUERIES_PATH if heldout else TEST_QUERIES_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single(
    query: dict,
    version: str,
    provider: str = "groq",
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> dict:
    """Run a single query against a single prompt version and evaluate."""
    from app import run_react_loop

    query_text = query["query"]
    expected_tool = query.get("expected_tool", "")
    category = query.get("category", "unknown")
    query_id = query.get("id", "")
    
    start = time.time()
    
    try:
        result = run_react_loop(
            user_message=query_text,
            chat_history=[],
            version=version,
            provider=provider,
            temperature=temperature,
            top_p=top_p,
            max_iterations=5,
        )
    except LLMUnavailableError as e:
        print(f"\n[FATAL] LLM provider unavailable: {e}")
        raise
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        # Score the error as a failed answer
        gt_case = GROUND_TRUTH_CASES.get(query_id, {
            "id": query_id, "expected_tool": expected_tool,
            "required": [], "forbidden": [], "policy_query": False,
        })
        scores = score_answer(f"ERROR: {str(e)[:100]}", [], gt_case, POLICY_FACTS)
        return {
            "query_id": query_id,
            "category": category,
            "version": version,
            "provider": provider,
            "task_success": scores["task_success"],
            "tool_accuracy": scores["tool_accuracy"],
            "hallucination": scores["hallucination"],
            "ungrounded": scores["ungrounded"],
            "correct": scores["correct"],
            "latency_ms": elapsed,
            "answer": f"ERROR: {str(e)[:100]}",
            "tools_called": "",
            "note": query.get("note", ""),
        }
    
    elapsed = result.get("total_latency_ms", round((time.time() - start) * 1000))
    retry_wait_ms = result.get("total_retry_wait_ms", 0)
    provider_used = result.get("provider", provider)
    model_used = result.get("model", "")
    fallback_used = int(bool(result.get("fallback_used", False)))
    tool_calls_valid_json = int(result.get("tool_calls_valid_json", 1))
    
    # Extract tools called
    tools_called = [step["action"] for step in result.get("trace", []) if step["action"] != "finish"]
    tools_called_str = ", ".join(tools_called)
    
    answer = result.get("answer", "")
    
    # --- Ground-truth scoring via scoring.py ---
    cases = load_ground_truth_cases()
    gt_case = cases.get(query_id, GROUND_TRUTH_CASES.get(query_id, {
        "id": query_id, "expected_tool": expected_tool,
        "required": [], "forbidden": [],
        "policy_query": category == "policy_rag",
    }))
    scores = score_answer(answer, tools_called, gt_case, POLICY_FACTS)
    
    return {
        "version": version,
        "query_id": query_id,
        "category": category,
        "provider": provider_used,
        "fallback_used": fallback_used,
        "model": model_used,
        "temperature": temperature,
        "top_p": top_p,
        "task_success": scores["task_success"],
        "tool_accuracy": scores["tool_accuracy"],
        "hallucination": scores["hallucination"],
        "ungrounded": scores["ungrounded"],
        "correct": scores["correct"],
        "latency_ms": elapsed,
        "retry_wait_ms": retry_wait_ms,
        "tool_calls_valid_json": tool_calls_valid_json,
        "tools_called": tools_called_str,
        "answer": answer[:200],  # Truncate for CSV
        "note": query.get("note", ""),
    }


RESULTS_FIELDNAMES = [
    "version", "query_id", "category", "provider", "fallback_used",
    "model", "temperature", "top_p", "task_success", "tool_accuracy",
    "hallucination", "ungrounded", "correct", "latency_ms",
    "retry_wait_ms", "tool_calls_valid_json", "tools_called", "answer", "note"
]


def run_full_evaluation(
    provider: str = "groq",
    versions: list[str] | None = None,
    temperature: float | None = None,
    delay_between: float = 1.0,
    heldout: bool = False,
    output_path: Path | None = None,
    queries_path: Path | None = None,
) -> list[dict]:
    import tools
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp()
    temp_db = Path(temp_dir) / "shopease_eval.db"
    orig_db = tools.DB_PATH
    if orig_db.exists():
        shutil.copy2(orig_db, temp_db)
    tools.DB_PATH = temp_db

    try:
        return _run_full_evaluation_isolated(
            provider, versions, temperature, delay_between, heldout, output_path, queries_path
        )
    finally:
        tools.DB_PATH = orig_db


def _run_full_evaluation_isolated(
    provider: str = "groq",
    versions: list[str] | None = None,
    temperature: float | None = None,
    delay_between: float = 1.0,
    heldout: bool = False,
    output_path: Path | None = None,
    queries_path: Path | None = None,
) -> list[dict]:
    """Run evaluation matrix across queries and versions.
    
    Saves incrementally after EVERY ROW to prevent data loss from quota errors.
    """
    queries = load_test_queries(heldout=heldout, queries_path=queries_path)
    all_versions = versions or list_versions()
    
    if output_path is None:
        output_path = HELDOUT_RESULTS_CSV if heldout else RESULTS_CSV_PATH
    
    # Check for existing completed rows so runs are resumable
    existing_keys = set()
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    existing_keys.add((r.get("version", "").strip(), r.get("query_id", "").strip()))
        except Exception:
            existing_keys = set()

    results = []
    total = len(queries) * len(all_versions)
    current = 0
    
    print("\n" + "="*60)
    print("ShopEase AI Evaluation")
    print(f"Provider: {provider} | Versions: {len(all_versions)} | Queries: {len(queries)}")
    print(f"Total evaluations: {total} | Existing completed: {len(existing_keys)}")
    if heldout:
        print("MODE: Held-out test set (no tuning!)")
    print("="*60 + "\n")
    
    for version in all_versions:
        ver_temp = temperature if temperature is not None else RECOMMENDED_TEMPS.get(version, 0.3)
        print(f"\n--- Version: {version} (T={ver_temp}) ---")
        version_results = []
        try:
            for query in queries:
                current += 1
                q_key = (version.strip(), query["id"].strip())
                if q_key in existing_keys:
                    print(f"  [{current}/{total}] {query['id']}: [ALREADY RUN] Skipping.")
                    continue

                print(f"  [{current}/{total}] {query['id']}: {query['query'][:50]}...", end=" ", flush=True)
                
                result = evaluate_single(query, version, provider, ver_temp)
                results.append(result)
                version_results.append(result)
                
                # Incremental save after every single row to never lose finished work
                merge_and_save_results([result], output_path)
                existing_keys.add(q_key)
                
                status = "[OK]" if result["task_success"] else "[X]"
                hall = " [HALLUC]" if result["hallucination"] else ""
                ungrounded = " [UNGROUNDED]" if result.get("ungrounded") else ""
                print(f"{status} ({result['latency_ms']}ms){hall}{ungrounded}")
                
                if delay_between > 0:
                    time.sleep(delay_between)
        except LLMUnavailableError:
            print(f"\n[HALTED] LLMUnavailableError encountered. Preserved {len(results)} rows in {output_path}.")
            break
        
        # Print per-version summary with Wilson intervals
        summary = summarize([
            {"task_success": r["task_success"], "tool_accuracy": r["tool_accuracy"],
             "hallucination": r["hallucination"], "ungrounded": r.get("ungrounded", 0)}
            for r in version_results
        ])
        print(f"  Summary: {summary}")
    
    return results


def save_results(results: list[dict], output_path: Path = RESULTS_CSV_PATH):
    """Save results to CSV and JSON."""
    if not results:
        print("No results to save.")
        return
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path} and {json_path}")
    print_summary(results)


def merge_and_save_results(results: list[dict], output_path: Path = RESULTS_CSV_PATH):
    """Save or merge new benchmark results into results.csv and results.json without wiping out other versions."""
    if not results:
        return
    
    existing_rows = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        except Exception:
            existing_rows = []
    
    merged_map = {(r.get("query_id"), r.get("version")): r for r in existing_rows}
    for r in results:
        merged_map[(r.get("query_id"), r.get("version"))] = r
    
    final_rows = list(merged_map.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, indent=2)


def print_summary(results: list[dict]):
    """Print a summary table of results by version with Wilson 95% CIs."""
    from collections import defaultdict
    
    version_stats = defaultdict(lambda: {"success": 0, "total": 0, "tool_acc": 0, "halluc": 0, "ungrounded": 0, "latency": []})
    
    for r in results:
        v = r["version"]
        version_stats[v]["total"] += 1
        version_stats[v]["success"] += int(r["task_success"])
        version_stats[v]["tool_acc"] += int(r["tool_accuracy"])
        version_stats[v]["halluc"] += int(r["hallucination"])
        version_stats[v]["ungrounded"] += int(r.get("ungrounded", 0))
        version_stats[v]["latency"].append(int(r["latency_ms"]))
    
    print("\n" + "="*100)
    print(f"{'Version':<25} {'Success%':>12} {'95% CI':>16} {'ToolAcc%':>10} {'Halluc%':>10} {'Ungrnd%':>10} {'AvgLatency':>12}")
    print("="*100)
    
    for v in sorted(version_stats.keys()):
        s = version_stats[v]
        n = s["total"]
        k_s = s["success"]
        success_pct = (k_s / n * 100) if n else 0
        lo, hi = wilson_interval(k_s, n)
        ci_str = f"({lo:.1%}-{hi:.1%})"
        tool_pct = (s["tool_acc"] / n * 100) if n else 0
        halluc_pct = (s["halluc"] / n * 100) if n else 0
        ungrnd_pct = (s["ungrounded"] / n * 100) if n else 0
        avg_lat = sum(s["latency"]) / len(s["latency"]) if s["latency"] else 0
        
        print(f"{v:<25} {success_pct:>9.1f}%   {ci_str:>16} {tool_pct:>9.1f}% {halluc_pct:>9.1f}% {ungrnd_pct:>9.1f}% {avg_lat:>10.0f}ms")
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Run ShopEase AI evaluation")
    parser.add_argument("--provider", default="groq", choices=["groq", "gemini", "openai", "openrouter", "anthropic"],
                       help="LLM provider (default: groq)")
    parser.add_argument("--versions", default=None,
                       help="Comma-separated version names to test (default: all)")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Temperature override (default: per-version recommended)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Seconds between API calls for rate limiting (default: 1.0)")
    parser.add_argument("--output", default=None,
                       help="Output CSV path")
    parser.add_argument("--heldout", action="store_true",
                       help="Run held-out test set on V6 only (no tuning)")
    parser.add_argument("--heldout-v2", action="store_true",
                       help="Run held-out test set v2 (with verified names) on V6 only")
    parser.add_argument("--queries-file", default=None,
                       help="Custom test queries JSON file")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs for spread analysis (default: 1)")
    
    args = parser.parse_args()
    
    # Strictly enforce fallback off for all evaluation runs
    import os
    os.environ["ALLOW_PROVIDER_FALLBACK"] = "0"
    
    versions = None
    if args.versions:
        requested = [v.strip() for v in args.versions.split(",")]
        canonical = list_versions()
        matched = []
        for r in requested:
            found = False
            for c in canonical:
                if r == c or r == c.split("—")[0].strip() or r.lower() == c.split("—")[0].strip().lower():
                    matched.append(c)
                    found = True
                    break
            if not found:
                matched.append(r)
        versions = matched
    
    # Held-out v2 mode: only V6, single run, no tuning
    if args.heldout_v2:
        versions = ["V6 — RAG + ReAct"]
        output = Path(args.output) if args.output else HELDOUT_V2_RESULTS_CSV
        print("\n*** HELD-OUT V2 EVALUATION (verified names) — run once, no tuning ***")
        results = run_full_evaluation(
            provider=args.provider,
            versions=versions,
            temperature=args.temperature,
            delay_between=args.delay,
            heldout=False,
            output_path=output,
            queries_path=HELDOUT_V2_QUERIES_PATH,
        )
        save_results(results, output)
        return

    # Held-out mode: only V6, single run, no tuning
    if args.heldout:
        versions = ["V6 — RAG + ReAct"]
        output = Path(args.output) if args.output else HELDOUT_RESULTS_CSV
        print("\n*** HELD-OUT EVALUATION — run once, no tuning ***")
        results = run_full_evaluation(
            provider=args.provider,
            versions=versions,
            temperature=args.temperature,
            delay_between=args.delay,
            heldout=True,
            output_path=output,
        )
        save_results(results, output)
        return
    
    queries_path = Path(args.queries_file) if args.queries_file else None
    output = Path(args.output) if args.output else RESULTS_CSV_PATH
    
    # Multi-run spread analysis
    if args.runs > 1:
        all_run_results = []
        for run_idx in range(1, args.runs + 1):
            print(f"\n{'#'*60}")
            print(f"  RUN {run_idx} of {args.runs}")
            print(f"{'#'*60}")
            run_results = run_full_evaluation(
                provider=args.provider,
                versions=versions,
                temperature=args.temperature,
                delay_between=args.delay,
                output_path=output.with_stem(f"{output.stem}_run{run_idx}"),
            )
            all_run_results.append(run_results)
        
        # Print cross-run spread
        print("\n" + "="*60)
        print("CROSS-RUN SPREAD ANALYSIS")
        print("="*60)
        from collections import defaultdict
        version_runs = defaultdict(list)
        for run_results in all_run_results:
            by_version = defaultdict(list)
            for r in run_results:
                by_version[r["version"]].append(r)
            for v, rows in by_version.items():
                k = sum(int(r["task_success"]) for r in rows)
                version_runs[v].append(k / len(rows) * 100)
        for v in sorted(version_runs.keys()):
            rates = version_runs[v]
            print(f"  {v}: {[f'{r:.1f}%' for r in rates]}  spread={max(rates)-min(rates):.1f}pp")
        return
    
    results = run_full_evaluation(
        provider=args.provider,
        versions=versions,
        temperature=args.temperature,
        delay_between=args.delay,
        output_path=output,
        queries_path=queries_path,
    )
    
    merge_and_save_results(results, output)


if __name__ == "__main__":
    main()
