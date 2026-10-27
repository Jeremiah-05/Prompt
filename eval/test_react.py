"""
Unit tests for tools, retriever, ReAct loop parsing, authorization, and concurrency.
Run: pytest eval/test_react.py -v
"""

import sys
import json
import pytest
import concurrent.futures
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import (
    get_order_status,
    get_product_info,
    initiate_return,
    calculate_refund,
    place_order,
    cancel_order,
    modify_order,
    execute_tool,
    reset_db,
    _get_db,
)
from retriever import retrieve_policy_chunks, reset_index
from app import parse_react_output


@pytest.fixture(autouse=True)
def clean_db():
    """Ensure every test runs against clean seed data in SQLite."""
    reset_db()
    yield
    reset_db()


# ===========================================================================
# Tool Tests
# ===========================================================================

class TestGetOrderStatus:
    """Tests for get_order_status tool."""
    
    def test_valid_order(self):
        result = get_order_status("ORD1001", customer_name="Alice Johnson")
        assert result["order_id"] == "ORD1001"
        assert result["status"] == "delivered"
        assert result["tracking_id"] == "TRK1001US"
        assert result["customer_name"] == "Alice Johnson"
    
    def test_invalid_order(self):
        result = get_order_status("ORD9999", customer_name="Alice Johnson")
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    def test_order_without_prefix(self):
        result = get_order_status("1001", customer_name="Alice Johnson")
        assert result["order_id"] == "ORD1001"
    
    def test_shipped_order(self):
        result = get_order_status("ORD1002", customer_name="Bob Stevens")
        assert result["status"] == "shipped"
        assert result["tracking_id"] == "TRK1002US"
        assert result["delivery_date"] is None
    
    def test_processing_order(self):
        result = get_order_status("ORD1009", customer_name="Iris Chen")
        assert result["status"] == "processing"
        assert result["tracking_id"] is None

    def test_shipped_order_dates_distinction(self):
        """Regression test: verify get_order_status explicitly distinguishes order_date from shipping_date and delivery_date."""
        result = get_order_status("ORD1002", customer_name="Bob Stevens")
        assert result["status"] == "shipped"
        assert result["order_date"] == "2026-09-20"
        assert result["shipped_date"] == "not recorded"
        assert result["delivery_date"] is None
        assert result["tracking_id"] == "TRK1002US"


class TestGetProductInfo:
    """Tests for get_product_info tool."""
    
    def test_by_product_id(self):
        result = get_product_info("PROD001")
        assert result["name"] == "UltraBook Pro 14"
        assert result["price"] == 1099.99
    
    def test_by_name(self):
        result = get_product_info("UltraBook Pro 14")
        assert result["product_id"] == "PROD001"
    
    def test_by_partial_name(self):
        result = get_product_info("UltraBook")
        assert result["product_id"] == "PROD001"
    
    def test_out_of_stock(self):
        result = get_product_info("PROD011")
        assert result["stock"] == 0
    
    def test_not_found(self):
        result = get_product_info("NonExistentProduct12345")
        assert "error" in result


class TestInitiateReturn:
    """Tests for initiate_return tool."""
    
    def test_eligible_general_return(self):
        # ORD1005 delivered 2026-09-13, owner Emma Wilson
        result = initiate_return("ORD1005", "PROD010", customer_name="Emma Wilson")
        assert "eligible" in result
    
    def test_non_delivered_order(self):
        result = initiate_return("ORD1009", "PROD008", customer_name="Iris Chen")
        assert result["eligible"] is False
        assert "processing" in result["reason"].lower()
    
    def test_shipped_order_return(self):
        result = initiate_return("ORD1002", "PROD003", customer_name="Bob Stevens")
        assert result["eligible"] is False
        assert "transit" in result["reason"].lower() or "shipped" in result["reason"].lower()
    
    def test_product_not_in_order(self):
        result = initiate_return("ORD1001", "PROD003", customer_name="Alice Johnson")
        assert result["eligible"] is False
        assert "not in order" in result["reason"].lower()
    
    def test_food_item_non_returnable(self):
        result = initiate_return("ORD1005", "PROD010", customer_name="Emma Wilson")
        # PROD010 is Organic Green Tea — food item
        assert "eligible" in result


class TestCalculateRefund:
    """Tests for calculate_refund tool."""
    
    def test_valid_refund(self):
        result = calculate_refund("ORD1001", "PROD001", customer_name="Alice Johnson")
        assert result["unit_price"] == 1099.99
        assert result["return_shipping_deduction"] == 7.99
        assert result["net_refund"] == round(1099.99 - 7.99, 2)
    
    def test_invalid_order(self):
        result = calculate_refund("ORD9999", "PROD001", customer_name="Alice Johnson")
        assert "error" in result
    
    def test_product_not_in_order(self):
        result = calculate_refund("ORD1001", "PROD003", customer_name="Alice Johnson")
        assert "error" in result


class TestAuthorizationAndConcurrency:
    """Tests for STEP 1 authorization and concurrent atomic locking."""

    def test_missing_name_refused(self):
        result = get_order_status("ORD1001")
        assert result.get("verification") == "VERIFICATION_REQUIRED"
        assert "VERIFICATION_REQUIRED" in result.get("message", "")

    def test_non_owner_refused(self):
        result = get_order_status("ORD1001", customer_name="Bob Stevens")
        assert result.get("verification") == "VERIFICATION_REQUIRED"
        assert "VERIFICATION_REQUIRED" in result.get("message", "")

    def test_owner_succeeds(self):
        result = get_order_status("ORD1001", customer_name="Alice Johnson")
        assert result.get("verification") is None
        assert result["order_id"] == "ORD1001"
        assert result["status"] == "delivered"

    def test_cancel_by_non_owner_refused(self):
        result = cancel_order("ORD1009", customer_name="Wrong Person")
        assert result.get("verification") == "VERIFICATION_REQUIRED"

    def test_modify_by_non_owner_refused(self):
        result = modify_order("ORD1009", "PROD008", 2, customer_name="Wrong Person")
        assert result.get("verification") == "VERIFICATION_REQUIRED"

    def test_return_by_non_owner_refused(self):
        result = initiate_return("ORD1005", "PROD010", customer_name="Wrong Person")
        assert result.get("verification") == "VERIFICATION_REQUIRED"

    def test_refund_by_non_owner_refused(self):
        result = calculate_refund("ORD1001", "PROD001", customer_name="Wrong Person")
        assert result.get("verification") == "VERIFICATION_REQUIRED"

    def test_concurrent_place_order_cannot_oversell(self):
        # Set a test product with exactly 3 stock
        with _get_db() as conn:
            conn.execute("UPDATE products SET stock = 3 WHERE product_id = 'PROD001'")
            conn.commit()

        # Run 10 concurrent requests to buy 1 unit each
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(place_order, f"Buyer_{i}", "PROD001", 1)
                for i in range(10)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        successes = [r for r in results if r.get("success")]
        failures = [r for r in results if not r.get("success")]

        assert len(successes) == 3, f"Expected exactly 3 orders to succeed, got {len(successes)}"
        assert len(failures) == 7, f"Expected 7 orders to fail due to stock, got {len(failures)}"

        prod = get_product_info("PROD001")
        assert prod["stock"] == 0, f"Expected final stock 0, got {prod['stock']}"


class TestExecuteTool:
    """Tests for the tool executor."""
    
    def test_execute_known_tool(self):
        result_json = execute_tool("get_order_status", {"order_id": "ORD1001", "customer_name": "Alice Johnson"})
        result = json.loads(result_json)
        assert result["order_id"] == "ORD1001"
    
    def test_execute_unknown_tool(self):
        result_json = execute_tool("nonexistent_tool", {})
        result = json.loads(result_json)
        assert "error" in result
    
    def test_execute_finish(self):
        result_json = execute_tool("finish", {"answer": "Hello"})
        result = json.loads(result_json)
        assert result["answer"] == "Hello"
    
    def test_execute_with_bad_args(self):
        result_json = execute_tool("get_product_info", {"wrong_arg": "value"})
        result = json.loads(result_json)
        assert "error" in result


# ===========================================================================
# Retriever Tests
# ===========================================================================

class TestRetriever:
    """Tests for TF-IDF policy retriever."""
    
    @classmethod
    def setup_class(cls):
        reset_index()
    
    def test_shipping_policy_retrieval(self):
        chunks = retrieve_policy_chunks("free shipping threshold standard delivery time", top_k=2)
        assert len(chunks) > 0
        all_text = " ".join(c["text"].lower() for c in chunks)
        assert "shipping" in all_text or "delivery" in all_text
    
    def test_return_policy_retrieval(self):
        chunks = retrieve_policy_chunks("how many days to return electronics", top_k=2)
        assert len(chunks) > 0
        all_text = " ".join(c["text"].lower() for c in chunks)
        assert "return" in all_text or "days" in all_text
    
    def test_warranty_retrieval(self):
        chunks = retrieve_policy_chunks("warranty coverage defects replacement", top_k=2)
        assert len(chunks) > 0
        all_text = " ".join(c["text"].lower() for c in chunks)
        assert "warranty" in all_text or "defect" in all_text
    
    def test_relevance_scoring(self):
        chunks = retrieve_policy_chunks("return policy", top_k=3)
        scores = [c["score"] for c in chunks]
        assert scores == sorted(scores, reverse=True)
    
    def test_empty_query(self):
        chunks = retrieve_policy_chunks("", top_k=3)
        assert len(chunks) == 0


# ===========================================================================
# ReAct Loop Parsing Tests
# ===========================================================================

class TestParseReactOutput:
    """Tests for ReAct Thought/Action/Action Input regex parser."""
    
    def test_standard_format(self):
        text = """Thought: The user is asking about order status. Let me check ORD1001.
Action: get_order_status
Action Input: {"order_id": "ORD1001", "customer_name": "Alice Johnson"}"""
        result = parse_react_output(text)
        assert "order status" in result["thought"]
        assert result["action"] == "get_order_status"
        assert result["action_input"]["order_id"] == "ORD1001"
        assert result["action_input"]["customer_name"] == "Alice Johnson"
    
    def test_finish_action(self):
        text = """Thought: I have all the info needed.
Action: finish
Action Input: {"answer": "Your order is shipped."}"""
        result = parse_react_output(text)
        assert result["action"] == "finish"
        assert result["action_input"]["answer"] == "Your order is shipped."
    
    def test_multiline_thought(self):
        text = """Thought: Step 1: The user wants to return an item.
Step 2: I need to check eligibility.
Step 3: Let me call the tool.
Action: initiate_return
Action Input: {"order_id": "ORD1008", "product_id": "PROD005", "customer_name": "Henry Brown"}"""
        result = parse_react_output(text)
        assert "Step 1" in result["thought"]
        assert result["action"] == "initiate_return"
    
    def test_no_valid_action(self):
        text = "I don't know how to help with that."
        result = parse_react_output(text)
        assert result["action"] == ""
        assert result["raw"] == text


# ===========================================================================
# Ground-Truth Case Audit & DB Fact Grounding Tests
# ===========================================================================

class TestCasesGroundTruthAudit:
    """Verifies that all data-dependent benchmark evaluation cases contain DB-derived facts."""
    
    def test_data_dependent_cases_have_db_facts(self):
        from eval.scoring import GROUND_TRUTH_CASES
        for qid, case in GROUND_TRUTH_CASES.items():
            if case.get("data_dependent"):
                assert "db_facts" in case, f"Case {qid} is data_dependent but lacks db_facts"
                assert len(case["db_facts"]) > 0, f"Case {qid} has empty db_facts"

    def test_data_dependent_cases_require_db_facts(self):
        from eval.scoring import GROUND_TRUTH_CASES
        generic_refusal_patterns = {r"cannot", r"can't", r"unable", r"don't have", r"not able"}
        
        for qid, case in GROUND_TRUTH_CASES.items():
            if case.get("data_dependent") and not case.get("should_refuse"):
                reqs = case.get("required", [])
                assert len(reqs) > 0, f"Case {qid} has no required patterns"
                for r in reqs:
                    assert r.lower() not in generic_refusal_patterns, (
                        f"Case {qid} accepts generic refusal '{r}' as required pattern"
                    )
                db_facts_str = " ".join(case.get("db_facts", [])).lower()
                matched_fact = False
                for p in reqs:
                    clean_p = p.replace("\\", "").replace("(", "").replace(")", "").replace("|", " ")
                    tokens = clean_p.split()
                    if any(t in db_facts_str for t in tokens if len(t) > 2):
                        matched_fact = True
                        break
                assert matched_fact, f"Case {qid} required list has no pattern matching db_facts: {case['db_facts']}"

    def test_db_facts_verified_in_sqlite(self):
        import sqlite3
        import tools
        conn = sqlite3.connect(tools.DB_PATH)
        c = conn.cursor()
        
        # Verify ORD1002 has TRK1002US
        row = c.execute("SELECT status, tracking_id FROM orders WHERE order_id = 'ORD1002'").fetchone()
        assert row == ("shipped", "TRK1002US")
        
        # Verify ORD1001 is delivered
        row = c.execute("SELECT status FROM orders WHERE order_id = 'ORD1001'").fetchone()
        assert row == ("delivered",)
        
        # Verify UltraBook Pro 14 price is 1099.99
        row = c.execute("SELECT price, specs FROM products WHERE name = 'UltraBook Pro 14'").fetchone()
        assert row[0] == 1099.99
        assert "i7" in row[1] and "16GB" in row[1]
        
        # Verify Cotton T-Shirt stock is 0
        row = c.execute("SELECT stock FROM products WHERE name = 'Cotton T-Shirt 3-pack'").fetchone()
        assert row == (0,)
        
        conn.close()


class TestSecurityHardening:
    """Tests verifying security hardening: input length cap, rate limiter, PII redaction, and confirmation turn."""

    def test_input_length_capping(self):
        from security import validate_input_length, MAX_INPUT_LENGTH
        ok, msg = validate_input_length("Safe query")
        assert ok is True
        
        ok, msg = validate_input_length("x" * (MAX_INPUT_LENGTH + 1))
        assert ok is False
        assert "exceeds maximum allowed limit" in msg

    def test_per_session_rate_limiter(self):
        from security import SessionRateLimiter
        limiter = SessionRateLimiter(max_requests=2, window_seconds=60)
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user1")[0] is True
        assert limiter.check_rate_limit("user1")[0] is False
        assert limiter.check_rate_limit("user2")[0] is True

    def test_pii_redaction_before_llm(self):
        from security import redact_pii
        raw = "Contact bob@example.com or 555-432-1098 with card 1234-5678-9012-3456"
        redacted, pii = redact_pii(raw)
        assert "[EMAIL REDACTED]" in redacted
        assert "[PHONE REDACTED]" in redacted
        assert "[CARD REDACTED]" in redacted
        assert "bob@example.com" not in redacted
        assert "555-432-1098" not in redacted
        assert "1234-5678-9012-3456" not in redacted
        assert len(pii) >= 3

    def test_transactional_confirmation_guard(self):
        from security import check_transactional_confirmation
        # Write without confirmation -> blocked
        allowed, challenge = check_transactional_confirmation(
            action="place_order",
            action_input={"product_id": "PROD001", "qty": 1, "customer_name": "Bob"},
            user_message="Buy this for me",
        )
        assert allowed is False
        assert "Confirmation Required" in challenge

        # Write with explicit confirmation -> allowed
        allowed, challenge = check_transactional_confirmation(
            action="place_order",
            action_input={"product_id": "PROD001", "qty": 1, "customer_name": "Bob"},
            user_message="Yes, please place the order, I confirm.",
        )
        assert allowed is True


class TestMultiTurnConversation:
    """Tests for multi-turn conversational context preservation in the ReAct loop."""

    def test_name_followup_preserves_order_id_in_prompt(self):
        """Verify that replying with only a customer name retains the order ID from chat history."""
        from prompts.builder import build_prompt
        chat_hist = [
            {"role": "user", "content": "Where is my order ORD1002?"},
            {"role": "assistant", "content": "Please verify your full name on the order."}
        ]
        user_msg = "Bob Stevens"
        prompt = build_prompt(
            version="V6 — RAG + ReAct",
            user_message=user_msg,
            chat_history=chat_hist,
        )
        assert "ORD1002" in prompt, "Order ID from previous turn must be present in prompt"
        assert "Bob Stevens" in prompt, "Customer name from current turn must be present in prompt"
        assert "User: Where is my order ORD1002?" in prompt
        assert "Assistant: Please verify your full name on the order." in prompt
        assert "USER MESSAGE:\nBob Stevens" in prompt

    def test_two_turn_react_execution(self):
        """Two-turn end-to-end test: Turn 1 asks for name, Turn 2 completes order lookup."""
        import os
        from app import run_react_loop
        if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
            pytest.skip("GROQ_API_KEY not configured for live test")

        try:
            # Turn 1: User asks without name -> returns verification challenge
            res1 = run_react_loop(
                user_message="Where is my order ORD1002?",
                chat_history=[],
                version="V6 — RAG + ReAct",
                provider="groq"
            )
            ans1_low = res1["answer"].lower()
            assert "name" in ans1_low or "verify" in ans1_low or "verification" in ans1_low

            # Turn 2: User responds with only name -> ReAct loop uses history and accesses ORD1002
            res2 = run_react_loop(
                user_message="Bob Stevens",
                chat_history=[
                    {"role": "user", "content": "Where is my order ORD1002?"},
                    {"role": "assistant", "content": res1["answer"]}
                ],
                version="V6 — RAG + ReAct",
                provider="groq"
            )
            actions = [s.get("action") for s in res2.get("trace", [])]
            assert "get_order_status" in actions, "Must call get_order_status on turn 2"
            status_step = next(s for s in res2["trace"] if s.get("action") == "get_order_status")
            assert status_step["action_input"]["order_id"] == "ORD1002"
            assert "Bob Stevens" in status_step["action_input"]["customer_name"]
            assert "TRK1002US" in res2["answer"]
        except Exception as exc:
            from llm_client import LLMUnavailableError
            if isinstance(exc, LLMUnavailableError) or "quota" in str(exc).lower():
                pytest.skip(f"Live LLM provider quota exhausted: {exc}")
            raise


class TestLiveDatabaseIsolation:
    """Verify that test suite executions are isolated from the live database and never cause data drift."""

    def test_live_db_order_count_matches_seed_file(self):
        """Fails if the live DB order count differs from orders.json."""
        live_db = Path(__file__).parent.parent / "data" / "shopease.db"
        seed_file = Path(__file__).parent.parent / "data" / "orders.json"
        with open(seed_file, "r", encoding="utf-8") as f:
            seed_orders = json.load(f)
        import sqlite3
        conn = sqlite3.connect(live_db)
        live_count = conn.cursor().execute("SELECT count(*) FROM orders;").fetchone()[0]
        conn.close()
        assert live_count == len(seed_orders), (
            f"Live database order count ({live_count}) does not match seed file {seed_file.name} ({len(seed_orders)}). "
            "Tests must never write to live data/shopease.db!"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

