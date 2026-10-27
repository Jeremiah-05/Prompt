import csv
import json
import statistics
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

print("=" * 80)
print("EVIDENCE-FIRST METRIC AUDIT & SOURCE FILE TRACE")
print("=" * 80 + "\n")

# 1. Main Benchmark: eval/results.csv
with open("eval/results.csv", "r", encoding="utf-8") as f:
    r_main = list(csv.DictReader(f))
v6_main = [r for r in r_main if "V6" in r["version"]]
succ_v6 = sum(int(r["task_success"]) for r in v6_main)
tool_v6 = sum(int(r["tool_accuracy"]) for r in v6_main)
hall_v6 = sum(int(r["hallucination"]) for r in v6_main)
ungr_v6 = sum(int(r.get("ungrounded", 0)) for r in v6_main)

print("SOURCE: eval/results.csv (Primary Benchmark, n=15)")
print(f"  - V6 Total Rows: {len(v6_main)}")
print(f"  - V6 Task Success: {succ_v6}/{len(v6_main)} ({succ_v6/len(v6_main)*100:.1f}%)")
print(f"  - V6 Tool Accuracy: {tool_v6}/{len(v6_main)} ({tool_v6/len(v6_main)*100:.1f}%)")
print(f"  - V6 Hallucination Rate: {hall_v6}/{len(v6_main)} ({hall_v6/len(v6_main)*100:.1f}%)")
print(f"  - V6 Ungrounded Rate: {ungr_v6}/{len(v6_main)} ({ungr_v6/len(v6_main)*100:.1f}%)\n")

# 2. Held-Out Evaluation: eval/heldout_results.csv
with open("eval/heldout_results.csv", "r", encoding="utf-8") as f:
    r_heldout = list(csv.DictReader(f))
lats_h = [float(r["latency_ms"]) for r in r_heldout]
succ_h = sum(int(r["task_success"]) for r in r_heldout)
tool_h = sum(int(r["tool_accuracy"]) for r in r_heldout)
hall_h = sum(int(r["hallucination"]) for r in r_heldout)
ungr_h = sum(int(r.get("ungrounded", 0)) for r in r_heldout)

print("SOURCE: eval/heldout_results.csv (Held-Out Benchmark, n=21, Groq Only, ALLOW_PROVIDER_FALLBACK=0)")
print(f"  - Total Rows: {len(r_heldout)}")
print(f"  - Providers: {set(r['provider'] for r in r_heldout)}")
print(f"  - Models: {set(r['model'] for r in r_heldout)}")
print(f"  - Latency min/med/max: {min(lats_h):.1f} / {statistics.median(lats_h):.1f} / {max(lats_h):.1f} ms")
print(f"  - Task Success: {succ_h}/{len(r_heldout)} ({succ_h/len(r_heldout)*100:.1f}%)")
print(f"  - Tool Accuracy: {tool_h}/{len(r_heldout)} ({tool_h/len(r_heldout)*100:.1f}%)")
print(f"  - Hallucination Rate: {hall_h}/{len(r_heldout)} ({hall_h/len(r_heldout)*100:.1f}%)")
print(f"  - Ungrounded Rate: {ungr_h}/{len(r_heldout)} ({ungr_h/len(r_heldout)*100:.1f}%)\n")

# 3. Bias Evaluation: eval/bias_results.csv
with open("eval/bias_results.csv", "r", encoding="utf-8") as f:
    r_bias = list(csv.DictReader(f))
print("SOURCE: eval/bias_results.csv (Fairness & Bias Robustness, 5 Pairs = 10 Queries, Groq Only)")
print(f"  - Total Rows: {len(r_bias)}")
print(f"  - Providers: {set(r['provider_used'] for r in r_bias)}")
pairs = {}
for r in r_bias:
    pairs[r["pair_id"]] = r["bias_equivalent"]
for p_id in sorted(pairs.keys()):
    print(f"  - {p_id}: {pairs[p_id]}")
print()

# 4. Temperature Sensitivity: eval/param_results.csv
with open("eval/param_results.csv", "r", encoding="utf-8") as f:
    r_param = list(csv.DictReader(f))
print("SOURCE: eval/param_results.csv (Temperature Sensitivity, n=45, Groq Only)")
print(f"  - Total Rows: {len(r_param)}")
for t_val in ["0.0", "0.3", "0.7"]:
    sub = [r for r in r_param if r["temperature"] == t_val]
    succ = sum(int(r["task_success"]) for r in sub)
    tool = sum(int(r["tool_accuracy"]) for r in sub)
    hall = sum(int(r["hallucination"]) for r in sub)
    lats = [float(r["latency_ms"]) for r in sub]
    print(f"  - T={t_val}: Success={succ}/{len(sub)} ({succ/len(sub)*100:.1f}%), ToolAcc={tool}/{len(sub)} ({tool/len(sub)*100:.1f}%), Halluc={hall}/{len(sub)} ({hall/len(sub)*100:.1f}%), MedLatency={statistics.median(lats):.0f}ms")
print()

# 5. Repeatability Runs: eval/results_run1.csv, results_run2.csv, results_run3.csv
for r_name in ["results_run1.csv", "results_run2.csv", "results_run3.csv"]:
    path = Path("eval") / r_name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            print(f"SOURCE: eval/{r_name} (Repeatability Run)")
            print(f"  - Total Preserved Rows: {len(rows)}")
            if rows:
                succ = sum(int(r["task_success"]) for r in rows)
                tool = sum(int(r["tool_accuracy"]) for r in rows)
                hall = sum(int(r["hallucination"]) for r in rows)
                print(f"  - Success: {succ}/{len(rows)} ({succ/len(rows)*100:.1f}%), ToolAcc: {tool}/{len(rows)}, Halluc: {hall}/{len(rows)}")
            else:
                print("  - Status: NOT RUN (0 rows) - provider daily quota exhausted")
print()

# 6. Database Counts: data/shopease.db
conn = sqlite3.connect("data/shopease.db")
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM orders")
orders_count = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM products")
products_count = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM telemetry_runs")
telemetry_count = c.fetchone()[0]
c.execute("SELECT COUNT(*), COUNT(CASE WHEN vote != '' AND vote IS NOT NULL THEN 1 END) FROM ab_log")
ab_total, ab_voted = c.fetchone()
print("SOURCE: data/shopease.db (SQLite Database)")
print(f"  - Orders table: {orders_count} rows")
print(f"  - Products table: {products_count} rows")
print(f"  - Telemetry Runs table: {telemetry_count} rows")
print(f"  - AB Log table: {ab_total} total comparisons, {ab_voted} human votes recorded")
print()

from telemetry import get_ab_win_report
print(f"A/B Win Report Output: {get_ab_win_report()}")
print("=" * 80)
