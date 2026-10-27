"""
Tools — 9 interactive functions backed by SQLite database (data/shopease.db).
These are called by the ReAct loop when the LLM outputs an Action.

Supports real-time CRUD with atomic transactions:
  - get_order_status   — look up existing order with ownership check
  - get_product_info   — look up product catalog
  - search_policy      — RAG-backed policy retrieval
  - initiate_return    — check return eligibility with ownership check
  - calculate_refund   — compute refund amounts with ownership check
  - place_order        — create a new order (atomic transaction, no oversell)
  - cancel_order       — cancel processing/shipped order with ownership check
  - modify_order       — change item quantity with ownership check
  - list_products      — browse the full product catalog
"""

import json
import random
import string
import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "shopease.db"


# ---------------------------------------------------------------------------
# SQLite Database Setup & Connection Helper
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    """Get a SQLite database connection with Row factory and WAL mode."""
    if not DB_PATH.exists():
        from data.seed_db import init_db
        init_db(DB_PATH)

    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def reset_db() -> None:
    """Reset the database to initial seed data."""
    from data.seed_db import init_db
    init_db(DB_PATH, force=True)


def _today() -> date:
    """Return today's date for eligibility checks."""
    return date.today()


def _generate_tracking_id() -> str:
    """Generate a unique tracking ID."""
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"TRK{suffix}IN"


def _row_to_order(row: sqlite3.Row) -> dict:
    """Convert an orders table SQLite row into a dict with parsed items."""
    d = dict(row)
    if isinstance(d.get("items"), str):
        try:
            d["items"] = json.loads(d["items"])
        except Exception:
            pass
    return d


def _row_to_product(row: sqlite3.Row) -> dict:
    """Convert a products table SQLite row into a dict."""
    return dict(row)


def _verify_owner(order: dict, customer_name: str | None, order_id: str) -> dict | None:
    """Deterministic ownership check shared by all order-accessing tools.

    Returns None if the caller is verified (name matches owner).
    Returns a VERIFICATION_REQUIRED dict otherwise.
    """
    if not customer_name or not customer_name.strip():
        return {
            "order_id": order_id,
            "verification": "VERIFICATION_REQUIRED",
            "error": "VERIFICATION_REQUIRED: customer does not match order owner. Do not disclose details.",
            "message": "VERIFICATION_REQUIRED: customer does not match order owner. Do not disclose details.",
        }
    owner = order["customer_name"]
    if customer_name.strip().lower() != owner.strip().lower():
        return {
            "order_id": order_id,
            "verification": "VERIFICATION_REQUIRED",
            "error": "VERIFICATION_REQUIRED: customer does not match order owner. Do not disclose details.",
            "message": "VERIFICATION_REQUIRED: customer does not match order owner. Do not disclose details.",
        }
    return None  # verified


# ---------------------------------------------------------------------------
# Core Tools
# ---------------------------------------------------------------------------

def get_order_status(order_id: str, customer_name: str | None = None) -> dict:
    """Look up order details by order ID with ownership verification.

    customer_name is checked against the order's owner. A missing or
    non-matching name returns VERIFICATION_REQUIRED so the model cannot
    override the access decision.

    Args:
        order_id: Order ID string (e.g., "ORD1001")
        customer_name: Caller name for ownership verification

    Returns:
        Dict with order details, verification failure, or error message.
    """
    order_id = order_id.strip().upper()
    if order_id.isdigit():
        order_id = f"ORD{order_id}"
    elif not order_id.startswith("ORD"):
        order_id = f"ORD{order_id}"

    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        row = cursor.fetchone()

    if not row:
        return {"error": f"Order '{order_id}' not found. Please verify the order ID and try again."}

    order = _row_to_order(row)

    # Deterministic ownership check
    denied = _verify_owner(order, customer_name, order_id)
    if denied:
        return denied

    return {
        "order_id": order["order_id"],
        "customer_name": order["customer_name"],
        "status": order["status"],
        "order_date": order["order_date"],
        "shipping_date": order.get("shipping_date", None),
        "delivery_date": order["delivery_date"],
        "tracking_id": order["tracking_id"],
        "items": order["items"],
        "total": order["total"],
        "payment_method": order["payment_method"],
        "return_eligible_until": order["return_eligible_until"],
    }


def get_product_info(product_id_or_name: str) -> dict:
    """Get product details by product ID or product name.

    Args:
        product_id_or_name: Product ID (e.g., "PROD001") or product name (partial match OK)

    Returns:
        Dict with product details or error message.
    """
    query = product_id_or_name.strip()

    def _format_prod(p: dict) -> dict:
        d = dict(p)
        d["currency"] = "INR"
        d["formatted_price"] = f"₹{p['price']}"
        return d

    with _get_db() as conn:
        cursor = conn.cursor()
        # Try exact product_id match first
        cursor.execute("SELECT * FROM products WHERE UPPER(product_id) = ?", (query.upper(),))
        row = cursor.fetchone()
        if row:
            return _format_prod(_row_to_product(row))

        # Try name match (case-insensitive partial)
        cursor.execute("SELECT * FROM products WHERE LOWER(name) LIKE ?", (f"%{query.lower()}%",))
        rows = cursor.fetchall()

    if len(rows) == 1:
        return _format_prod(_row_to_product(rows[0]))
    elif len(rows) > 1:
        matches = [_row_to_product(r) for r in rows]
        return {
            "multiple_matches": [
                {"product_id": p["product_id"], "name": p["name"], "price": p["price"], "stock": p["stock"]}
                for p in matches
            ],
            "message": f"Found {len(matches)} products matching '{query}'. Please specify a product ID.",
        }

    return {"error": f"Product '{query}' not found. Please check the product name or ID."}


def search_policy(query: str) -> dict:
    """Search store policy using RAG retrieval (delegates to retriever module).

    Args:
        query: Policy search query

    Returns:
        Dict with retrieved policy chunks and scores.
    """
    from retriever import retrieve_policy_chunks

    chunks = retrieve_policy_chunks(query, top_k=3)

    return {
        "query": query,
        "chunks": [{"text": c["text"], "score": round(c["score"], 4)} for c in chunks],
        "num_results": len(chunks),
    }


def initiate_return(order_id: str, product_id: str, customer_name: str | None = None) -> dict:
    """Check return eligibility and initiate a return.

    Validates:
    - Caller owns the order (customer_name check)
    - Order exists and is delivered
    - Product is in the order
    - Return window hasn't expired (30 days general, 14 days electronics)
    - Item is returnable (not food/perishable)

    Args:
        order_id: Order ID
        product_id: Product ID to return
        customer_name: Caller name for ownership verification

    Returns:
        Dict with eligibility status and message.
    """
    order_id = order_id.strip().upper()
    product_id = product_id.strip().upper()

    if not order_id.startswith("ORD"):
        order_id = f"ORD{order_id}"
    if not product_id.startswith("PROD"):
        product_id = f"PROD{product_id}"

    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        order_row = cursor.fetchone()

        if not order_row:
            return {"eligible": False, "reason": f"Order '{order_id}' not found."}

        order = _row_to_order(order_row)

        # Ownership verification
        denied = _verify_owner(order, customer_name, order_id)
        if denied:
            return denied

        # Check order is delivered
        if order["status"] != "delivered":
            if order["status"] == "cancelled":
                return {"eligible": False, "reason": f"Order {order_id} has already been cancelled."}
            elif order["status"] == "processing":
                return {"eligible": False, "reason": f"Order {order_id} is still processing. You can cancel it instead of returning."}
            elif order["status"] == "shipped":
                return {"eligible": False, "reason": f"Order {order_id} is in transit. Please wait for delivery before initiating a return."}

        # Check product is in order
        order_product_ids = [item["product_id"] for item in order["items"]]
        if product_id not in order_product_ids:
            return {
                "eligible": False,
                "reason": f"Product {product_id} is not in order {order_id}. Items in this order: {order_product_ids}",
            }

        # Look up product details
        cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
        prod_row = cursor.fetchone()
        if not prod_row:
            return {"eligible": False, "reason": f"Product {product_id} not found in catalog."}

        product = _row_to_product(prod_row)

    # Check if food/perishable (non-returnable)
    category = product.get("category", "")
    if category == "grocery":
        non_returnable_keywords = ["tea", "coffee", "chocolate", "food"]
        if any(kw in product["name"].lower() for kw in non_returnable_keywords):
            return {
                "eligible": False,
                "reason": f"{product['name']} is a food/perishable item and is non-returnable unless damaged during shipping. Report within 48 hours with photos.",
            }

    # Check return window
    today = _today()
    if order.get("return_eligible_until"):
        eligible_until = datetime.strptime(order["return_eligible_until"], "%Y-%m-%d").date()
        if today > eligible_until:
            days_expired = (today - eligible_until).days
            return {
                "eligible": False,
                "reason": f"Return window expired on {order['return_eligible_until']} ({days_expired} days ago). Standard return window: 30 days for general items, 14 days for electronics.",
            }
        days_remaining = (eligible_until - today).days
        return {
            "eligible": True,
            "order_id": order_id,
            "product_id": product_id,
            "product_name": product["name"],
            "return_eligible_until": order["return_eligible_until"],
            "days_remaining": days_remaining,
            "customer_name": order["customer_name"],
            "message": f"Return approved for {product['name']} (Order {order_id}). Return window open until {order['return_eligible_until']} ({days_remaining} days remaining). A prepaid return shipping label has been sent to your email.",
        }

    return {
        "eligible": False,
        "reason": f"Order {order_id} does not have a return eligibility date set (status: {order['status']}).",
    }


def calculate_refund(order_id: str, product_id: str, customer_name: str | None = None) -> dict:
    """Calculate the refund amount for a returned product.

    Args:
        order_id: Order ID
        product_id: Product ID
        customer_name: Caller name for ownership verification

    Returns:
        Dict with refund amount, deductions, and net refund.
    """
    order_id = order_id.strip().upper()
    product_id = product_id.strip().upper()

    if not order_id.startswith("ORD"):
        order_id = f"ORD{order_id}"
    if not product_id.startswith("PROD"):
        product_id = f"PROD{product_id}"

    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        order_row = cursor.fetchone()

        if not order_row:
            return {"error": f"Order '{order_id}' not found."}

        order = _row_to_order(order_row)

        # Ownership verification
        denied = _verify_owner(order, customer_name, order_id)
        if denied:
            return denied

        # Check product is in order
        order_product_ids = [item["product_id"] for item in order["items"]]
        if product_id not in order_product_ids:
            return {"error": f"Product {product_id} is not in order {order_id}."}

        cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
        prod_row = cursor.fetchone()
        if not prod_row:
            return {"error": f"Product {product_id} not found."}

        product = _row_to_product(prod_row)

    # Get quantity from order
    qty = 1
    for item in order["items"]:
        if item["product_id"] == product_id:
            qty = item.get("qty", 1)
            break

    item_total = round(product["price"] * qty, 2)
    return_shipping = 7.99
    net_refund = round(item_total - return_shipping, 2)

    return {
        "order_id": order_id,
        "product_id": product_id,
        "quantity": qty,
        "unit_price": product["price"],
        "item_total": item_total,
        "currency": "INR",
        "formatted_item_total": f"₹{item_total}",
        "return_shipping_deduction": return_shipping,
        "net_refund": net_refund,
        "formatted_net_refund": f"₹{net_refund}",
        "refund_method": order.get("payment_method", "original payment method"),
        "estimated_timeline": "5-7 business days after item received",
    }


# ---------------------------------------------------------------------------
# Order CRUD Operations (with Atomic SQLite Transactions)
# ---------------------------------------------------------------------------

def place_order(customer_name: str, product_id: str, qty: int = 1, payment_method: str = "credit_card") -> dict:
    """Place a new order for a product from the catalog.

    Creates a real order in the system with status 'processing'.
    Validates stock availability and deducts inventory atomically via SQLite transaction.

    Args:
        customer_name: Name of the customer placing the order
        product_id: Product ID to order (e.g., "PROD001")
        qty: Quantity to order (default 1)
        payment_method: Payment method — credit_card, debit_card, paypal, upi (default credit_card)

    Returns:
        Dict with new order confirmation or error.
    """
    product_id = product_id.strip().upper()
    if not product_id.startswith("PROD"):
        product_id = f"PROD{product_id}"

    customer_name = customer_name.strip() if customer_name else ""
    if not customer_name:
        return {"error": "Customer name is required to place an order."}

    qty = int(qty) if qty else 1
    if qty < 1:
        return {"error": "Quantity must be at least 1."}
    if qty > 10:
        return {"error": "Maximum 10 units per order."}

    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")

        cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
        prod_row = cursor.fetchone()
        if not prod_row:
            conn.rollback()
            return {"error": f"Product '{product_id}' not found."}

        product = _row_to_product(prod_row)

        if product["stock"] < qty:
            conn.rollback()
            if product["stock"] == 0:
                return {
                    "error": f"Sorry, {product['name']} is currently out of stock.",
                    "product": product["name"],
                    "stock": 0,
                }
            return {
                "error": f"Insufficient stock for {product['name']}. Available: {product['stock']}, Requested: {qty}.",
                "product": product["name"],
                "available_stock": product["stock"],
            }

        # Atomic stock deduction with stock condition
        cursor.execute(
            "UPDATE products SET stock = stock - ? WHERE product_id = ? AND stock >= ?",
            (qty, product_id, qty),
        )
        if cursor.rowcount == 0:
            conn.rollback()
            return {"error": f"Insufficient stock for {product['name']}."}

        # Determine next available order number
        cursor.execute("SELECT order_id FROM orders")
        rows = cursor.fetchall()
        existing_nums = []
        for r in rows:
            try:
                existing_nums.append(int(r["order_id"].replace("ORD", "")))
            except ValueError:
                pass
        next_order_num = max(existing_nums, default=1020) + 1
        order_id = f"ORD{next_order_num}"

        today_str = _today().isoformat()
        total = round(product["price"] * qty, 2)
        shipping_fee = 0 if total >= 499 else 49
        order_total = round(total + shipping_fee, 2)
        items_json = json.dumps([{"product_id": product_id, "qty": qty}])

        cursor.execute(
            """
            INSERT INTO orders (
                order_id, customer_name, items, status, order_date,
                delivery_date, tracking_id, return_eligible_until, payment_method, total
            )
            VALUES (?, ?, ?, 'processing', ?, NULL, NULL, NULL, ?, ?)
            """,
            (order_id, customer_name, items_json, today_str, payment_method, order_total),
        )

        conn.commit()

    estimated_delivery = (_today() + timedelta(days=5)).isoformat()

    return {
        "success": True,
        "order_id": order_id,
        "customer_name": customer_name,
        "product": product["name"],
        "product_id": product_id,
        "quantity": qty,
        "unit_price": product["price"],
        "subtotal": total,
        "shipping_fee": shipping_fee,
        "order_total": order_total,
        "currency": "INR",
        "formatted_total": f"₹{order_total}",
        "payment_method": payment_method,
        "status": "processing",
        "order_date": today_str,
        "estimated_delivery": estimated_delivery,
        "message": (
            f"🎉 Order {order_id} placed successfully! {product['name']} x{qty} for ₹{order_total}. "
            f"Estimated delivery by {estimated_delivery}. You will receive tracking info once shipped."
        ),
    }


def cancel_order(order_id: str, customer_name: str | None = None) -> dict:
    """Cancel an existing order.

    Only orders in 'processing' or 'shipped' status can be cancelled.
    Processing orders get a full refund. Shipped orders get a refund minus return shipping.
    Stock is restored on cancellation atomically via SQLite transaction.

    Args:
        order_id: Order ID to cancel (e.g., "ORD1001")
        customer_name: Caller name for ownership verification

    Returns:
        Dict with cancellation confirmation or error.
    """
    order_id = order_id.strip().upper()
    if order_id.isdigit():
        order_id = f"ORD{order_id}"
    elif not order_id.startswith("ORD"):
        order_id = f"ORD{order_id}"

    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")

        cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        order_row = cursor.fetchone()
        if not order_row:
            conn.rollback()
            return {"error": f"Order '{order_id}' not found."}

        order = _row_to_order(order_row)

        denied = _verify_owner(order, customer_name, order_id)
        if denied:
            conn.rollback()
            return denied

        status = order["status"]
        if status == "cancelled":
            conn.rollback()
            return {"error": f"Order {order_id} has already been cancelled."}

        if status == "delivered":
            conn.rollback()
            return {
                "error": f"Order {order_id} has already been delivered and cannot be cancelled. "
                         f"Please use the return process instead (initiate_return tool)."
            }

        refund_amount = order["total"]
        shipping_deduction = 0
        if status == "shipped":
            shipping_deduction = 79.0
            refund_amount = round(order["total"] - shipping_deduction, 2)

        # Restore stock for each item
        restored_items = []
        for item in order["items"]:
            item_prod_id = item["product_id"]
            qty = item.get("qty", 1)
            cursor.execute(
                "UPDATE products SET stock = stock + ? WHERE product_id = ?",
                (qty, item_prod_id),
            )
            cursor.execute("SELECT name FROM products WHERE product_id = ?", (item_prod_id,))
            p_row = cursor.fetchone()
            p_name = p_row["name"] if p_row else item_prod_id
            restored_items.append({"product_id": item_prod_id, "name": p_name, "qty_restored": qty})

        cursor.execute("UPDATE orders SET status = 'cancelled' WHERE order_id = ?", (order_id,))
        conn.commit()

    return {
        "success": True,
        "order_id": order_id,
        "previous_status": status,
        "new_status": "cancelled",
        "refund_amount": refund_amount,
        "formatted_refund": f"₹{refund_amount}",
        "shipping_deduction": shipping_deduction,
        "refund_method": order.get("payment_method", "original payment method"),
        "estimated_refund_timeline": "5-7 business days",
        "restored_stock": restored_items,
        "message": (
            f"✅ Order {order_id} has been cancelled. Refund of ₹{refund_amount} will be "
            f"credited to your {order.get('payment_method', 'account').replace('_', ' ')} "
            f"within 5-7 business days."
            + (f" ₹{shipping_deduction} shipping deduction applied (order was already shipped)." if shipping_deduction else "")
        ),
    }


def modify_order(order_id: str, product_id: str, new_qty: int, customer_name: str | None = None) -> dict:
    """Modify the quantity of a product in an existing order.

    Only orders in 'processing' status can be modified. Adjusts stock atomically.

    Args:
        order_id: Order ID to modify (e.g., "ORD1021")
        product_id: Product ID to change quantity for
        new_qty: New quantity (must be >= 1)
        customer_name: Caller name for ownership verification

    Returns:
        Dict with modification details or error.
    """
    order_id = order_id.strip().upper()
    product_id = product_id.strip().upper()

    if order_id.isdigit():
        order_id = f"ORD{order_id}"
    elif not order_id.startswith("ORD"):
        order_id = f"ORD{order_id}"
    if not product_id.startswith("PROD"):
        product_id = f"PROD{product_id}"

    new_qty = int(new_qty) if new_qty else 1
    if new_qty < 1:
        return {"error": "Quantity must be at least 1. To remove the item entirely, cancel the order."}
    if new_qty > 10:
        return {"error": "Maximum 10 units per item per order."}

    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")

        cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        order_row = cursor.fetchone()
        if not order_row:
            conn.rollback()
            return {"error": f"Order '{order_id}' not found."}

        order = _row_to_order(order_row)

        denied = _verify_owner(order, customer_name, order_id)
        if denied:
            conn.rollback()
            return denied

        if order["status"] != "processing":
            conn.rollback()
            return {
                "error": f"Order {order_id} has status '{order['status']}' and cannot be modified. "
                         f"Only orders in 'processing' status can be changed."
            }

        # Find target item
        target_item = None
        for item in order["items"]:
            if item["product_id"] == product_id:
                target_item = item
                break

        if not target_item:
            conn.rollback()
            return {"error": f"Product {product_id} is not in order {order_id}."}

        old_qty = target_item.get("qty", 1)
        qty_diff = new_qty - old_qty

        cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
        prod_row = cursor.fetchone()
        if not prod_row:
            conn.rollback()
            return {"error": f"Product {product_id} not found in catalog."}

        product = _row_to_product(prod_row)

        if qty_diff > 0:
            cursor.execute(
                "UPDATE products SET stock = stock - ? WHERE product_id = ? AND stock >= ?",
                (qty_diff, product_id, qty_diff),
            )
            if cursor.rowcount == 0:
                conn.rollback()
                return {
                    "error": f"Insufficient stock to increase quantity. Available: {product['stock']}, Need: {qty_diff} more.",
                    "current_qty": old_qty,
                    "requested_qty": new_qty,
                }
        elif qty_diff < 0:
            cursor.execute(
                "UPDATE products SET stock = stock + ? WHERE product_id = ?",
                (abs(qty_diff), product_id),
            )

        # Update items
        target_item["qty"] = new_qty

        # Recalculate order total
        new_total = 0.0
        for item in order["items"]:
            cursor.execute("SELECT price FROM products WHERE product_id = ?", (item["product_id"],))
            p_row = cursor.fetchone()
            if p_row:
                new_total += p_row["price"] * item.get("qty", 1)

        shipping_fee = 0 if new_total >= 499 else 49
        order_total = round(new_total + shipping_fee, 2)
        old_total = order["total"]

        cursor.execute(
            "UPDATE orders SET items = ?, total = ? WHERE order_id = ?",
            (json.dumps(order["items"]), order_total, order_id),
        )

        conn.commit()

    return {
        "success": True,
        "order_id": order_id,
        "product_id": product_id,
        "product_name": product["name"],
        "old_qty": old_qty,
        "new_qty": new_qty,
        "old_total": old_total,
        "new_total": order_total,
        "formatted_new_total": f"₹{order_total}",
        "price_difference": round(order_total - old_total, 2),
        "message": (
            f"📝 Order {order_id} updated: {product['name']} quantity changed from {old_qty} to {new_qty}. "
            f"New order total: ₹{order_total}."
        ),
    }


def list_products(category: str = "") -> dict:
    """Browse the full product catalog, optionally filtered by category.

    Args:
        category: Optional category filter — 'electronics', 'apparel', or 'grocery'.
                  Leave empty to show all products.

    Returns:
        Dict with list of products and availability.
    """
    category = category.strip().lower() if category else ""

    if category:
        valid_cats = ["electronics", "apparel", "grocery"]
        if category not in valid_cats:
            return {"error": f"Unknown category '{category}'. Available categories: {valid_cats}"}

    with _get_db() as conn:
        cursor = conn.cursor()
        if category:
            cursor.execute("SELECT * FROM products WHERE LOWER(category) = ?", (category,))
        else:
            cursor.execute("SELECT * FROM products")
        rows = cursor.fetchall()

    result_items = []
    for r in rows:
        p = _row_to_product(r)
        result_items.append({
            "product_id": p["product_id"],
            "name": p["name"],
            "category": p.get("category", ""),
            "price": p["price"],
            "formatted_price": f"₹{p['price']}",
            "stock": p["stock"],
            "in_stock": p["stock"] > 0,
        })

    return {
        "total_products": len(result_items),
        "filter": category if category else "all",
        "products": result_items,
    }


# ---------------------------------------------------------------------------
# Tool Registry — All 9 tools
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_order_status": get_order_status,
    "get_product_info": get_product_info,
    "search_policy": search_policy,
    "initiate_return": initiate_return,
    "calculate_refund": calculate_refund,
    "place_order": place_order,
    "cancel_order": cancel_order,
    "modify_order": modify_order,
    "list_products": list_products,
}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name with given input and return JSON string result.

    Args:
        tool_name: Name of the tool to execute
        tool_input: Dict of arguments for the tool

    Returns:
        JSON string of tool result
    """
    if tool_name == "finish":
        return json.dumps(tool_input)

    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}"})

    try:
        func = TOOL_REGISTRY[tool_name]
        result = func(**tool_input)
        return json.dumps(result, default=str)
    except TypeError as e:
        return json.dumps({"error": f"Invalid arguments for {tool_name}: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Tool execution error: {str(e)}"})
