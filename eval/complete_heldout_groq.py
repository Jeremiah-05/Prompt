import os
import sys
import time
from pathlib import Path

# Force Groq only, strictly no fallback
os.environ["ALLOW_PROVIDER_FALLBACK"] = "0"
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from eval.run_eval import (
    load_test_queries,
    evaluate_single,
    merge_and_save_results,
    HELDOUT_RESULTS_CSV,
)
import csv

all_heldout = load_test_queries(heldout=True)

# Find already completed IDs
completed_ids = set()
if HELDOUT_RESULTS_CSV.exists():
    with open(HELDOUT_RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("provider") == "groq" and float(row.get("latency_ms", 0)) > 50:
                completed_ids.add(row["query_id"])

remaining = [q for q in all_heldout if q["id"] not in completed_ids]

print(f"Total held-out queries: {len(all_heldout)}")
print(f"Already completed on Groq: {len(completed_ids)} ({sorted(completed_ids)})")
rem_ids = [q['id'] for q in remaining]
print(f"Remaining to run: {len(remaining)} ({rem_ids})")

version = "V6 — RAG + ReAct"
provider = "groq"
temperature = 0.3

for idx, query in enumerate(remaining, 1):
    q_id = query["id"]
    print(f"\n[{idx}/{len(remaining)}] Running {q_id}: {query['query'][:50]}...", flush=True)
    res = evaluate_single(query, version, provider, temperature)
    merge_and_save_results([res], HELDOUT_RESULTS_CSV)
    status = "[OK]" if res["task_success"] else "[X]"
    print(f"  -> {status} (ToolAcc: {res['tool_accuracy']}, Halluc: {res['hallucination']}, Latency: {res['latency_ms']}ms)")
    print(f"     Answer: {res['answer'][:100]}")
    time.sleep(3.5)

print("\n" + "=" * 60)
print("Held-out evaluation completed successfully!")
print("=" * 60)
