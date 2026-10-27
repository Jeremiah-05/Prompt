"""
Seed SQLite database data/shopease.db from data/orders.json and data/products.json.
Maintains JSON files as seed data only.
"""

import json
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "shopease.db"
ORDERS_JSON = DATA_DIR / "orders.json"
PRODUCTS_JSON = DATA_DIR / "products.json"


def init_db(db_path: Path = DB_PATH, force: bool = False) -> None:
    """Initialize SQLite database schema and seed from JSON files."""
    if db_path.exists() and not force:
        return

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()

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

    # Seed products
    if PRODUCTS_JSON.exists():
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

    # Seed orders
    if ORDERS_JSON.exists():
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
    conn.close()
    print(f"Initialized and seeded {db_path} successfully.")


if __name__ == "__main__":
    init_db(force=True)
