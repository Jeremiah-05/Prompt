import sqlite3
from pathlib import Path

db = Path("data/shopease.db")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

orders_cnt = conn.execute("SELECT count(*) as c FROM orders").fetchone()["c"]
products_cnt = conn.execute("SELECT count(*) as c FROM products").fetchone()["c"]
telemetry_cnt = conn.execute("SELECT count(*) as c FROM telemetry_runs").fetchone()["c"]
ab_log_cnt = conn.execute("SELECT count(*) as c FROM ab_log").fetchone()["c"]
ab_voted_cnt = conn.execute("SELECT count(*) as c FROM ab_log WHERE vote IS NOT NULL AND vote != ''").fetchone()["c"]
ab_votes = conn.execute("SELECT vote, count(*) as c FROM ab_log GROUP BY vote").fetchall()

print(f"ORDERS: {orders_cnt}")
print(f"PRODUCTS: {products_cnt}")
print(f"TELEMETRY_RUNS: {telemetry_cnt}")
print(f"AB_LOG_TOTAL: {ab_log_cnt}")
print(f"AB_LOG_VOTED: {ab_voted_cnt}")
for r in ab_votes:
    print(f"  vote: {r['vote']} -> {r['c']}")
conn.close()
