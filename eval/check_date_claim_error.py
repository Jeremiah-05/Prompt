import csv
import json
import re
import sqlite3
from pathlib import Path

DB_PATH = Path("data/shopease.db")
RESULTS_CSV = Path("eval/results.csv")

def get_orders():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM orders").fetchall()
    orders = {r["order_id"]: dict(r) for r in rows}
    conn.close()
    return orders

def check_date_claim_error(answer: str, orders: dict) -> int:
    """
    Checks if a reply attaches an order date to 'shipped on' or 'delivered on'
    when the database has no ship date (orders table has only order_date and delivery_date)
    or delivery_date is null/different.
    E.g. 'shipped on 2026-09-20' for ORD1002 where 2026-09-20 is order_date, not shipped date.
    Or 'delivered on <date>' when delivery_date is null or mismatches.
    """
    if not answer:
        return 0
    
    # Check for mentions of orders
    order_matches = re.findall(r"ORD\d{4}", answer, re.IGNORECASE)
    
    # Check for patterns like 'shipped on <date>' or 'delivered on <date>'
    shipped_match = re.search(r"shipped\s+on\s+(\d{4}-\d{2}-\d{2})", answer, re.IGNORECASE)
    delivered_match = re.search(r"delivered\s+on\s+(\d{4}-\d{2}-\d{2})", answer, re.IGNORECASE)
    
    for oid in order_matches:
        order = orders.get(oid.upper())
        if not order:
            continue
        order_date = order.get("order_date")
        delivery_date = order.get("delivery_date")
        
        # If bot says 'shipped on <date>' and date matches order_date (database has no ship_date column, only order_date)
        if shipped_match:
            claimed_date = shipped_match.group(1)
            if claimed_date == order_date:
                return 1
                
        # If bot says 'delivered on <date>' when delivery_date is null or doesn't match
        if delivered_match:
            claimed_date = delivered_match.group(1)
            if not delivery_date or claimed_date != delivery_date:
                return 1
                
    # Also check general phrasing if no specific ORD match in regex but date is attached
    if shipped_match:
        # If claimed date is 2026-09-20 (ORD1002 order date)
        if shipped_match.group(1) == "2026-09-20":
            return 1

    return 0

def audit():
    orders = get_orders()
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    counts = {}
    flagged = []
    for r in rows:
        v = r["version"]
        ans = r.get("answer", "")
        err = check_date_claim_error(ans, orders)
        counts[v] = counts.get(v, 0) + err
        if err:
            flagged.append((v, r["query_id"], ans[:120]))
            
    print("Date claim errors per version in eval/results.csv:")
    for v, c in counts.items():
        print(f"  {v}: {c}/15")
    print(f"Total flagged: {len(flagged)}")
    for f in flagged:
        print(f"  {f[0]} | {f[1]} | {f[2]}")

if __name__ == "__main__":
    audit()
