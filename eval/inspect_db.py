import sqlite3
import json

conn = sqlite3.connect('data/shopease.db')
c = conn.cursor()
tables = [row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('TABLES:', tables)

for t in tables:
    cols = [col[1] for col in c.execute(f"PRAGMA table_info({t})").fetchall()]
    print(f"\n--- {t} (columns: {cols}) ---")
    rows = c.execute(f"SELECT * FROM {t}").fetchall()
    print(f"Total rows: {len(rows)}")
    for r in rows[:3]:
        print(" ", r)

# Also let's inspect the specific orders and products referenced in Q01-Q15:
# Q01: ORD1002 (Bob Stevens)
# Q02: ORD9999 (fake)
# Q03: ORD1001 (Alice Johnson)
# Q04: UltraBook Pro 14 (PRD001 or similar)
# Q05: Cotton T-Shirt 3-pack (PRD005 or similar)
# Q06: SmartPhone X12, 4K Smart TV
# Q07: ORD1008 (Henry Brown), Denim Jacket
# Q08: ORD1001 (Alice Johnson), UltraBook Pro 14
# Q09: ORD1004 (David Lee), SmartPhone X12
# Q10: policy (shipping)
# Q11: policy (refunds)
# Q12: policy (warranty)
# Q13: fake order ORD9999
# Q14: ORD1001 (Alice Johnson), 90-day extension
# Q15: recommend most expensive product

print("\n--- SPECIFIC ORDERS ---")
for oid in ['ORD1001', 'ORD1002', 'ORD1004', 'ORD1008', 'ORD9999']:
    r = c.execute("SELECT * FROM orders WHERE order_id = ?", (oid,)).fetchall()
    print(oid, r)

print("\n--- SPECIFIC PRODUCTS ---")
for name in ['UltraBook%', 'Cotton%', 'SmartPhone%', '4K Smart TV%']:
    r = c.execute("SELECT * FROM products WHERE name LIKE ?", (name,)).fetchall()
    print(name, r)
