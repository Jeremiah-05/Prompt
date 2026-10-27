# -*- coding: utf-8 -*-
"""
eval/test_cases.py
Unit tests verifying that all data-dependent benchmark evaluation cases
contain DB-derived facts and do not accept generic refusals as success.
"""

import json
import re
import sqlite3
from pathlib import Path
import pytest

EVAL_DIR = Path(__file__).parent
CASES_PATH = EVAL_DIR / "cases.json"
DB_PATH = EVAL_DIR.parent / "data" / "shopease.db"


def load_cases():
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_data_dependent_cases_have_db_facts():
    """Every data-dependent case must have a non-empty db_facts list."""
    cases = load_cases()
    for case in cases:
        if case.get("data_dependent"):
            assert "db_facts" in case, f"Case {case['id']} is data_dependent but lacks 'db_facts'"
            assert len(case["db_facts"]) > 0, f"Case {case['id']} has empty 'db_facts'"


def test_data_dependent_cases_require_db_facts():
    """
    Data-dependent cases where should_refuse is False must contain at least one
    real DB-derived fact in 'required' and must NOT accept generic refusals.
    """
    cases = load_cases()
    generic_refusal_patterns = {r"cannot", r"can't", r"unable", r"don't have", r"not able"}
    
    for case in cases:
        if case.get("data_dependent") and not case.get("should_refuse"):
            reqs = case.get("required", [])
            assert len(reqs) > 0, f"Case {case['id']} has no required patterns"
            
            # Ensure not solely composed of generic refusal words
            for r in reqs:
                assert r.lower() not in generic_refusal_patterns, (
                    f"Case {case['id']} accepts generic refusal '{r}' as required pattern"
                )
            
            # Check that at least one required pattern corresponds to a db_fact
            db_facts_str = " ".join(case.get("db_facts", [])).lower()
            matched_fact = False
            for p in reqs:
                clean_p = p.replace("\\", "").replace("(", "").replace(")", "").replace("|", " ")
                tokens = clean_p.split()
                if any(t in db_facts_str for t in tokens if len(t) > 2):
                    matched_fact = True
                    break
            assert matched_fact, f"Case {case['id']} required list has no pattern matching its db_facts: {case['db_facts']}"


def test_db_facts_exist_in_database():
    """Verify that facts referenced in db_facts actually exist in data/shopease.db."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Q01: ORD1002 has TRK1002US and status shipped
    row = c.execute("SELECT status, tracking_id FROM orders WHERE order_id = 'ORD1002'").fetchone()
    assert row == ("shipped", "TRK1002US")
    
    # Q03: ORD1001 is delivered
    row = c.execute("SELECT status FROM orders WHERE order_id = 'ORD1001'").fetchone()
    assert row == ("delivered",)
    
    # Q04: UltraBook Pro 14 price is 1099.99
    row = c.execute("SELECT price, specs FROM products WHERE name = 'UltraBook Pro 14'").fetchone()
    assert row[0] == 1099.99
    assert "i7" in row[1] and "16GB" in row[1]
    
    # Q05: Cotton T-Shirt 3-pack stock is 0
    row = c.execute("SELECT stock FROM products WHERE name = 'Cotton T-Shirt 3-pack'").fetchone()
    assert row == (0,)
    
    conn.close()


def test_failing_case_without_db_facts():
    """Demonstrate that an invalid data-dependent case with no DB fact fails the check."""
    invalid_case = {
        "id": "Q_BAD",
        "data_dependent": True,
        "should_refuse": False,
        "required": ["cannot", "unable"],
        "db_facts": []
    }
    with pytest.raises(AssertionError):
        assert len(invalid_case["db_facts"]) > 0


def test_regression_order_date_not_confused_with_shipping_date():
    """
    Regression test: verify that date_claim_error correctly flags replies
    that falsely claim order_date is the shipping date for orders where ship date is unrecorded.
    """
    from eval.check_date_claim_error import check_date_claim_error, get_orders
    orders = get_orders()

    # Reply incorrectly claiming the order was shipped on 2026-09-20 (its placement date)
    bad_reply = "Your order ORD1002 was shipped on 2026-09-20. Tracking ID: TRK1002US."
    assert check_date_claim_error(bad_reply, orders) == 1

    # Accurate reply specifying placement date without fabricating shipment date
    good_reply = "Your order ORD1002 is shipped! Tracking ID is TRK1002US. It was placed on 2026-09-20."
    assert check_date_claim_error(good_reply, orders) == 0

    # Reply explicitly noting shipment date is unrecorded
    good_reply_unrecorded = "Your order ORD1002 is shipped. Specific shipment date is not recorded in our database; order date: 2026-09-20."
    assert check_date_claim_error(good_reply_unrecorded, orders) == 0


def test_scoring_forbidden_pattern_shipped_on_order_date():
    """
    Regression test: verify eval/scoring.py flags 'shipped on <order_date>' as hallucination for Q01.
    """
    from eval.scoring import score_answer, GROUND_TRUTH_CASES
    q01 = GROUND_TRUTH_CASES["Q01"]
    
    # Model incorrectly claims order was shipped on 2026-09-20
    bad_reply = "Your order ORD1002 was shipped on 2026-09-20. Tracking: TRK1002US."
    res_bad = score_answer(bad_reply, ["get_order_status"], q01, set())
    assert res_bad["hallucination"] == 1, "Must flag 'shipped on <order_date>' as hallucination"
    assert res_bad["task_success"] == 0

    # Model correctly gives status and tracking without claiming order_date as shipping date
    good_reply = "Your order ORD1002 has shipped with tracking TRK1002US. Order date: 2026-09-20, shipped date: not recorded."
    res_good = score_answer(good_reply, ["get_order_status"], q01, set())
    assert res_good["hallucination"] == 0
    assert res_good["task_success"] == 1


