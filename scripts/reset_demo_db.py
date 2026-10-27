"""
Reset Demo Database (ShopEase AI)
Drop and reseed data/shopease.db from data/orders.json and data/products.json,
preserving telemetry_runs and ab_log tables.
Prints total order count and per-product stock, verifying Demo Script alignment.
"""

import os
import sys
import json
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from eval.live_tracker import clear_live_logs

DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "shopease.db"
ORDERS_JSON = DATA_DIR / "orders.json"
PRODUCTS_JSON = DATA_DIR / "products.json"


def reset_demo_db() -> dict:
    assert ORDERS_JSON.exists(), f"Missing {ORDERS_JSON}"
    assert PRODUCTS_JSON.exists(), f"Missing {PRODUCTS_JSON}"

    # Clear live interaction logs to reset live tracker stream
    clear_live_logs()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()

    # Verify or initialize telemetry tables if needed so they are preserved
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL,
            version TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            user_query TEXT NOT NULL,
            final_answer TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            est_cost_usd REAL NOT NULL,
            tool_calls TEXT,
            demo_mode INTEGER DEFAULT 0
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ab_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            version_a TEXT NOT NULL,
            temp_a REAL NOT NULL,
            version_b TEXT NOT NULL,
            temp_b REAL NOT NULL,
            latency_a_ms INTEGER NOT NULL,
            tokens_a INTEGER NOT NULL,
            answer_a TEXT NOT NULL,
            trace_a TEXT,
            latency_b_ms INTEGER NOT NULL,
            tokens_b INTEGER NOT NULL,
            answer_b TEXT NOT NULL,
            trace_b TEXT,
            vote TEXT
        );
    """)

    # Drop and recreate ONLY orders and products
    cursor.execute("DROP TABLE IF EXISTS products;")
    cursor.execute("DROP TABLE IF EXISTS orders;")

    cursor.execute("""
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL,
            specs TEXT,
            warranty TEXT
        );
    """)

    cursor.execute("""
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL,
            items TEXT NOT NULL,
            status TEXT NOT NULL,
            order_date TEXT NOT NULL,
            delivery_date TEXT,
            tracking_id TEXT,
            return_eligible_until TEXT,
            payment_method TEXT,
            total REAL NOT NULL
        );
    """)

    # Reseed products
    with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
        products = json.load(f)
    for p in products:
        cursor.execute(
            """
            INSERT INTO products (product_id, name, category, price, stock, specs, warranty)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p["product_id"],
                p["name"],
                p["category"],
                float(p["price"]),
                int(p["stock"]),
                p.get("specs", ""),
                p.get("warranty", ""),
            ),
        )

    # Reseed orders
    with open(ORDERS_JSON, "r", encoding="utf-8") as f:
        orders = json.load(f)
    for o in orders:
        items_json = json.dumps(o["items"]) if isinstance(o["items"], (list, dict)) else o["items"]
        cursor.execute(
            """
            INSERT INTO orders (
                order_id, customer_name, items, status, order_date,
                delivery_date, tracking_id, return_eligible_until, payment_method, total
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                o["order_id"],
                o["customer_name"],
                items_json,
                o["status"],
                o["order_date"],
                o.get("delivery_date"),
                o.get("tracking_id"),
                o.get("return_eligible_until"),
                o.get("payment_method"),
                float(o["total"]),
            ),
        )

    conn.commit()

    # Query counts & verify
    order_count = cursor.execute("SELECT count(*) FROM orders;").fetchone()[0]
    prod_rows = cursor.execute("SELECT product_id, name, stock FROM products ORDER BY product_id;").fetchall()
    telemetry_count = cursor.execute("SELECT count(*) FROM telemetry_runs;").fetchone()[0]
    ab_count = cursor.execute("SELECT count(*) FROM ab_log;").fetchone()[0]

    # Verify Demo Script alignments
    ord1001 = cursor.execute("SELECT customer_name, status, delivery_date FROM orders WHERE order_id = 'ORD1001';").fetchone()
    ord1002 = cursor.execute("SELECT customer_name, status, tracking_id, order_date FROM orders WHERE order_id = 'ORD1002';").fetchone()
    ord1008 = cursor.execute("SELECT customer_name, status, delivery_date FROM orders WHERE order_id = 'ORD1008';").fetchone()

    conn.close()

    print("================================================================================")
    print("RESET DEMO DATABASE COMPLETE")
    print("================================================================================")
    print(f"Target Database: {DB_PATH}")
    print(f"Orders Seeded: {order_count} (from {ORDERS_JSON.name})")
    print(f"Preserved Telemetry Runs: {telemetry_count}")
    print(f"Preserved A/B Logs: {ab_count}")
    print("Live Interaction Logs: Cleared (live_logs.json)")
    print("\nPer-Product Stock Levels:")
    print("-" * 55)
    for pid, pname, stock in prod_rows:
        print(f"  {pid:7} | {pname:32} | stock: {stock:3}")
    print("-" * 55)

    print("\nDemo Script Ground-Truth Verification:")
    print(f"  [Demo Step 1] ORD1002: Owner={ord1002[0]} (Bob Stevens), Status={ord1002[1]}, Tracking={ord1002[2]}")
    print(f"  [Demo Step 3] ORD1001: Owner={ord1001[0]} (Alice Johnson), Status={ord1001[1]}, DeliveryDate={ord1001[2]}")
    print(f"  [Demo Step 4] ORD1008: Owner={ord1008[0]} (Henry Brown), Status={ord1008[1]}, DeliveryDate={ord1008[2]}")

    assert ord1001[0] == "Alice Johnson", f"ORD1001 owner mismatch: {ord1001[0]}"
    assert ord1002[0] == "Bob Stevens" and ord1002[2] == "TRK1002US", f"ORD1002 mismatch: {ord1002}"
    assert ord1008[0] == "Henry Brown", f"ORD1008 owner mismatch: {ord1008[0]}"

    return {
        "orders_count": order_count,
        "products_count": len(prod_rows),
        "telemetry_count": telemetry_count,
        "ab_count": ab_count,
    }


if __name__ == "__main__":
    reset_demo_db()
