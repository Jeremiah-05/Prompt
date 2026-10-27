import csv

with open("eval/results.csv", "r", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r.get("version") == "V1 — Naive"]

print(f"Total V1 rows: {len(rows)}")
for r in rows:
    qid = r.get("id") or r.get("query_id")
    prov = r.get("provider")
    ts = r.get("timestamp")
    succ = r.get("task_success")
    halluc = r.get("hallucination")
    ungr = r.get("ungrounded")
    ans_snippet = (r.get("answer") or "")[:70].replace("\n", " ")
    print(f"id={qid:<4} | prov={prov} | ts={ts} | succ={succ} | halluc={halluc} | ungr={ungr} | ans={ans_snippet}")
