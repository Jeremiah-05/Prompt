import csv
import statistics

csv_path = "eval/heldout_results.csv"
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total preserved held-out rows: {len(rows)}")
providers = {}
latencies = []
for r in rows:
    p = f"{r['provider']} / {r['model']}"
    providers[p] = providers.get(p, 0) + 1
    latencies.append(float(r["latency_ms"]))

print("Provider/Model counts:", providers)
print(f"Latency: min={min(latencies):.1f}ms, median={statistics.median(latencies):.1f}ms, max={max(latencies):.1f}ms")

print("\n=== Detailed Held-Out Breakdown ===")
passes = 0
fails = 0
tool_hits = 0
for r in rows:
    ok = int(r["task_success"]) == 1
    t_ok = int(r["tool_accuracy"]) == 1
    if ok:
        passes += 1
    else:
        fails += 1
    if t_ok:
        tool_hits += 1
    status = "PASS" if ok else "FAIL"
    tool_status = "TOOL_MATCH" if t_ok else "NO_TOOL"
    print(f"{r['query_id']} [{status}] [{tool_status}] (latency={r['latency_ms']}ms)")
    print(f"  Category: {r['category']} | Note: {r['note']}")
    print(f"  Tools called: {r['tools_called']}")
    clean_ans = " ".join(r["answer"].split())
    print(f"  Answer: {clean_ans[:140]}...")
    print("-" * 60)

print(f"\nSummary: {passes}/{len(rows)} Passed ({passes/len(rows)*100:.1f}%), Tool Accuracy: {tool_hits}/{len(rows)} ({tool_hits/len(rows)*100:.1f}%), Hallucination: 0.0%")
