"""
Generate V1-V6 x Q01-Q15 per-query matrix from eval/results.csv,
and audit all passed queries for V1 and V2 against DB facts and required refusals.
"""

import csv
import json
from pathlib import Path

WORKSPACE = Path(r"e:\New folder")
RESULTS_CSV = WORKSPACE / "eval" / "results.csv"
CASES_JSON = WORKSPACE / "eval" / "cases.json"

with open(RESULTS_CSV, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

with open(CASES_JSON, "r", encoding="utf-8") as f:
    cases = {c["id"]: c for c in json.load(f)}

versions = ["V1 — Naive", "V2 — Role + Rules", "V3 — Few-Shot", "V4 — CoT", "V5 — ReAct", "V6 — RAG + ReAct"]
qids = [f"Q{i:02d}" for i in range(1, 16)]

matrix = {}
for q in qids:
    matrix[q] = {}

for r in rows:
    v = r["version"]
    q = r["query_id"]
    if v in versions and q in qids:
        succ = int(r.get("task_success", 0))
        hall = int(r.get("hallucination", 0))
        tool = int(r.get("tool_accuracy", 0))
        if succ:
            status = "PASS"
        elif hall:
            status = "HALLUCINATION"
        elif not tool:
            status = "NO-TOOL"
        else:
            status = "WRONG"
        matrix[q][v] = status

print("==========================================================================================")
print("PER-QUERY EVALUATION MATRIX (V1–V6 x Q01–Q15)")
print("==========================================================================================")
header = f"| Query ID | Description | " + " | ".join(v.split("—")[0].strip() for v in versions) + " |"
separator = "|:---|:---|" + "|".join(":---:" for _ in versions) + "|"
print(header)
print(separator)

for q in qids:
    c = cases.get(q, {})
    desc = c.get("note", c.get("category", ""))[:32]
    row_str = f"| **{q}** | {desc} | " + " | ".join(matrix[q].get(v, "-") for v in versions) + " |"
    print(row_str)

print("\n==========================================================================================")
print("V1 & V2 PASSED QUERIES AUDIT (Ground-Truth DB Facts / Refusals)")
print("==========================================================================================")
for target_v in ["V1 — Naive", "V2 — Role + Rules"]:
    print(f"\n--- {target_v} ---")
    v_rows = [r for r in rows if r["version"] == target_v]
    for r in v_rows:
        if int(r.get("task_success", 0)) == 1:
            q = r["query_id"]
            c = cases.get(q, {})
            req = c.get("required", [])
            facts = c.get("db_facts", [])
            refuse = c.get("should_refuse", False)
            ans = r.get("answer", "").strip().replace("\n", " ")
            print(f"[{q}] ({c.get('category')}) Refusal={refuse} | Facts={facts}")
            print(f"     Required Regex: {req}")
            print(f"     Answer text: \"{ans[:160]}...\"")
