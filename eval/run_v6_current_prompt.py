# -*- coding: utf-8 -*-
"""
Run V6 on Q01-Q15 using current compact prompt definitions on Groq with fallback off.
Saves to eval/results_v6_current_prompt.csv.
"""
import sys
import os
import json
import csv
import time
import hashlib
from pathlib import Path

# Ensure fallback is strictly disabled
os.environ["ALLOW_PROVIDER_FALLBACK"] = "0"
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import run_react_loop
from llm_client import LLMUnavailableError
from eval.scoring import score_answer, load_policy_facts, GROUND_TRUTH_CASES

EVAL_DIR = Path(__file__).parent
OUTPUT_CSV = EVAL_DIR / "results_v6_current_prompt.csv"
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"

POLICY_FACTS = load_policy_facts()

FIELDNAMES = [
    "version", "query_id", "category", "provider", "fallback_used",
    "model", "temperature", "task_success", "tool_accuracy",
    "hallucination", "ungrounded", "correct", "latency_ms",
    "retry_wait_ms", "prompt_hash", "tools_called", "answer", "note"
]

def load_queries():
    with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run():
    queries = load_queries()
    
    # Check already completed queries for resume
    completed = set()
    rows = []
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed.add(r["query_id"])
                rows.append(r)
        print(f"Resuming from {len(completed)} completed queries.")

    version_label = "V6 — RAG + ReAct (current prompt)"

    for q in queries:
        qid = q["id"]
        if qid in completed:
            continue
        
        print(f"Running {qid} on Groq ({version_label})...", flush=True)
        start = time.time()
        try:
            res = run_react_loop(
                user_message=q["query"],
                chat_history=[],
                version="V6 — RAG + ReAct",
                provider="groq",
                temperature=0.3,
                max_iterations=5,
            )
            elapsed = res.get("total_latency_ms", round((time.time() - start) * 1000))
            retry_wait = res.get("total_retry_wait_ms", 0)
            
            full_prompt = res.get("full_prompt_sent", "")
            p_hash = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()[:12] if full_prompt else "N/A"
            
            tools_called = [step["action"] for step in res.get("trace", []) if step["action"] != "finish"]
            
            gt_case = GROUND_TRUTH_CASES.get(qid, {
                "id": qid, "expected_tool": q.get("expected_tool", ""),
                "required": [], "forbidden": [],
                "policy_query": q.get("category") == "policy_rag"
            })
            
            ans = res.get("answer", "")
            scores = score_answer(ans, tools_called, gt_case, POLICY_FACTS)
            
            row = {
                "version": version_label,
                "query_id": qid,
                "category": q.get("category", ""),
                "provider": res.get("provider", "groq"),
                "fallback_used": int(bool(res.get("fallback_used", False))),
                "model": res.get("model", "qwen/qwen3.8-27b"),
                "temperature": 0.3,
                "task_success": scores["task_success"],
                "tool_accuracy": scores["tool_accuracy"],
                "hallucination": scores["hallucination"],
                "ungrounded": scores["ungrounded"],
                "correct": scores["correct"],
                "latency_ms": elapsed,
                "retry_wait_ms": retry_wait,
                "prompt_hash": p_hash,
                "tools_called": ", ".join(tools_called),
                "answer": ans[:200],
                "note": q.get("note", "")
            }
            rows.append(row)
            
            # Write immediately to file
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
                
            print(f"  -> {qid} Done: success={scores['task_success']}, lat={elapsed}ms", flush=True)
            
            # Short sleep to respect Groq free tier rate limits (6000 TPM)
            time.sleep(2.0)

        except LLMUnavailableError as e:
            print(f"\n[FATAL] LLM provider unavailable on {qid}: {e}", flush=True)
            print(f"Preserving {len(rows)} finished rows. Marking remaining as NOT RUN.", flush=True)
            break
        except Exception as e:
            print(f"\n[ERROR] Exception on {qid}: {e}", flush=True)
            break

    print(f"\nCompleted {len(rows)}/15 queries in {OUTPUT_CSV}")

if __name__ == "__main__":
    run()
