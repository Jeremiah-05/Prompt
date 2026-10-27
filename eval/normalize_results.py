import csv
import json
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT_DIR / "eval"
BACKUP_DIR = EVAL_DIR / "_purged_backup"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

results_csv = EVAL_DIR / "results.csv"
results_json = EVAL_DIR / "results.json"

# Backup first
shutil.copy2(results_csv, BACKUP_DIR / "results_before_step2_normalize.csv")
if results_json.exists():
    shutil.copy2(results_json, BACKUP_DIR / "results_before_step2_normalize.json")
print("Backed up results.csv to eval/_purged_backup/results_before_step2_normalize.csv")

with open(results_csv, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows before normalization: {len(rows)}")

CANONICAL_VERSIONS = [
    "V1 — Naive",
    "V2 — Role + Rules",
    "V3 — Few-Shot",
    "V4 — CoT",
    "V5 — ReAct",
    "V6 — RAG + ReAct",
]

# Map aliases or remove known bad evaluation error rows
normalized_rows = []
removed_rows = []

for r in rows:
    v = r.get("version", "").strip()
    ans = r.get("answer", "")
    
    # Check for the bogus unknown prompt version rows
    if "Unknown prompt version" in ans and v == "V4 — Chain-of-Thought":
        removed_rows.append(r)
        continue
    
    # Normalize version naming if needed
    if v in ("V4 - Chain-of-Thought", "V4 — Chain-of-Thought") and "Unknown prompt version" not in ans:
        r["version"] = "V4 — CoT"
    elif v in ("V1 - Naive", "V1 -- Naive"):
        r["version"] = "V1 — Naive"
    elif v in ("V2 - Role + Rules", "V2 -- Role + Rules"):
        r["version"] = "V2 — Role + Rules"
    elif v in ("V3 - Few-Shot", "V3 -- Few-Shot"):
        r["version"] = "V3 — Few-Shot"
    elif v in ("V4 - CoT", "V4 -- CoT"):
        r["version"] = "V4 — CoT"
    elif v in ("V5 - ReAct", "V5 -- ReAct"):
        r["version"] = "V5 — ReAct"
    elif v in ("V6 - RAG + ReAct", "V6 -- RAG + ReAct"):
        r["version"] = "V6 — RAG + ReAct"
    
    normalized_rows.append(r)

print(f"Removed bogus error rows: {len(removed_rows)}")
print(f"Total rows after normalization: {len(normalized_rows)}")

# Verify per-version counts
from collections import Counter
counts = Counter(r["version"] for r in normalized_rows)
print("\nPer-version row counts:")
for cv in CANONICAL_VERSIONS:
    print(f"  {cv}: {counts[cv]} rows")

# Write back normalized results
fieldnames = list(rows[0].keys())
with open(results_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(normalized_rows)

with open(results_json, "w", encoding="utf-8") as f:
    json.dump(normalized_rows, f, indent=2)

print("\nSuccessfully updated eval/results.csv and eval/results.json!")
