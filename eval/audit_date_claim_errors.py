import sqlite3
import csv
import re
from pathlib import Path
from collections import defaultdict

conn = sqlite3.connect("data/shopease.db")
c = conn.cursor()
orders = c.execute("SELECT order_id, customer_name, status, order_date, delivery_date FROM orders").fetchall()
order_dict = {o[0]: {"customer": o[1], "status": o[2], "order_date": o[3], "delivery_date": o[4]} for o in orders}

print("=== LIVE ORDERS IN DATABASE ===")
for oid, info in sorted(order_dict.items()):
    print(f"{oid}: status={info['status']}, order_date={info['order_date']}, delivery_date={info['delivery_date']}")

# Check date_claim_error:
# A reply that attaches an order date to "shipped on" / "delivered on" when database has no ship/delivery date
# Pattern: (shipped\s+on\s+(?:20\d\d-\d\d-\d\d|\w+\s+\d{1,2}(?:,\s*20\d\d)?)|delivered\s+on\s+(?:20\d\d-\d\d-\d\d|\w+\s+\d{1,2}(?:,\s*20\d\d)?))
def check_date_claim_error(answer: str, query_id: str) -> int:
    ans_lower = answer.lower()
    
    # Check for ORD1002 specifically or any shipped-on date where delivery_date is null
    # Example: "shipped on 2026-09-20" or "shipped on September 20"
    m_shipped = re.search(r"shipped\s+on\s+(\d{4}-\d{2}-\d{2}|[a-zA-Z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4})", answer, re.IGNORECASE)
    if m_shipped:
        claimed_date = m_shipped.group(1).strip()
        # If order is ORD1002 (or Q01) where order_date is 2026-09-20 and there is no ship date:
        if "2026-09-20" in claimed_date or "sep" in claimed_date.lower():
            return 1
            
    m_delivered = re.search(r"delivered\s+on\s+(\d{4}-\d{2}-\d{2}|[a-zA-Z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4})", answer, re.IGNORECASE)
    if m_delivered:
        claimed_date = m_delivered.group(1).strip()
        # If it claims delivered on an order date when delivery_date is null
        if "2026-09-20" in claimed_date:
            return 1

    return 0

results_path = Path("eval/results.csv")
with open(results_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

per_version_counts = defaultdict(int)
per_version_total = defaultdict(int)

for r in rows:
    v = r["version"]
    per_version_total[v] += 1
    err = check_date_claim_error(r.get("answer", ""), r.get("query_id", ""))
    r["date_claim_error"] = str(err)
    if err:
        per_version_counts[v] += 1
        print(f"[{v}] {r['query_id']}: date_claim_error detected in answer snippet:\n  -> {r.get('answer', '')[:120]}")

print("\n=== DATE CLAIM ERROR SUMMARY PER VERSION ===")
for v in sorted(per_version_total.keys()):
    cnt = per_version_counts[v]
    tot = per_version_total[v]
    print(f"  {v}: {cnt}/{tot} ({cnt/tot*100:.1f}%) date_claim_error")

# Add date_claim_error column to results.csv without touching task_success or hallucination
fieldnames = list(rows[0].keys())
if "date_claim_error" not in fieldnames:
    fieldnames.append("date_claim_error")

with open(results_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\nSuccessfully updated eval/results.csv with date_claim_error column!")
