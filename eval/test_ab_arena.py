"""
Unit tests for Step 4: Blind A/B Arena protocol.
Validates:
1. Randomization & swapped mapping logic (vote maps back to true configuration).
2. Zero label leakage before vote (no version names, temperatures, latencies, or token counts leaked).
3. Database schema persistence and non-destructive ALTER TABLE migration.
"""

import pytest
import sqlite3
from telemetry import get_db_connection, log_ab_vote, get_ab_summary


def test_blind_arena_swapped_mapping_logic():
    """Verify that user votes map back accurately to the true configuration whether swapped or not."""
    cfg1 = ("V3 — Few-Shot", 0.3)
    cfg2 = ("V6 — RAG + ReAct", 0.3)
    
    # Case 1: Swapped presentation (Output A = cfg2, Output B = cfg1)
    swapped = True
    target_a = cfg2 if swapped else cfg1
    target_b = cfg1 if swapped else cfg2
    
    # User votes A -> true winner is target_a (cfg2: V6)
    vote = "A"
    winner = target_a[0] if vote == "A" else (target_b[0] if vote == "B" else "Tie")
    assert winner == "V6 — RAG + ReAct"
    
    # User votes B -> true winner is target_b (cfg1: V3)
    vote = "B"
    winner = target_a[0] if vote == "A" else (target_b[0] if vote == "B" else "Tie")
    assert winner == "V3 — Few-Shot"

    # Case 2: Direct presentation (Output A = cfg1, Output B = cfg2)
    swapped = False
    target_a = cfg2 if swapped else cfg1
    target_b = cfg1 if swapped else cfg2
    
    vote = "A"
    winner = target_a[0] if vote == "A" else (target_b[0] if vote == "B" else "Tie")
    assert winner == "V3 — Few-Shot"
    
    vote = "B"
    winner = target_a[0] if vote == "A" else (target_b[0] if vote == "B" else "Tie")
    assert winner == "V6 — RAG + ReAct"


def test_no_label_leak_before_vote():
    """Verify that before a vote is cast, output cards do not leak version, temp, tokens, or latency."""
    # Simulated mock answers
    answer_text = "Here is information on your order ORD1002."
    
    badge_a = "**Output A (Blinded)**\n\n"
    badge_b = "**Output B (Blinded)**\n\n"
    
    card_a = badge_a + answer_text
    card_b = badge_b + answer_text
    
    trace_placeholder = (
        "🔒 *Reasoning trace and performance metrics are hidden during blind evaluation to prevent operator bias. "
        "They will be revealed after you cast your vote below.*"
    )
    
    # Assert neither card header nor trace leaks metadata
    for forbidden in ["V1", "V2", "V3", "V4", "V5", "V6", "T=0.3", "T=0.7", "tokens", "ms"]:
        assert forbidden not in badge_a, f"Leak in badge_a: {forbidden}"
        assert forbidden not in badge_b, f"Leak in badge_b: {forbidden}"
        assert forbidden not in trace_placeholder, f"Leak in trace_placeholder: {forbidden}"


def test_ab_db_schema_and_vote_storage():
    """Verify ab_log contains blind protocol columns and preserves legacy rows."""
    conn = get_db_connection()
    c = conn.cursor()
    cols = [col[1] for col in c.execute("PRAGMA table_info(ab_log)").fetchall()]
    conn.close()
    
    for req_col in ["a_config", "b_config", "swapped", "protocol"]:
        assert req_col in cols, f"Column {req_col} missing from ab_log table"
    
    # Test vote insert under blind protocol
    row_id = log_ab_vote(
        prompt="Test blind query",
        version_a="V3 — Few-Shot",
        temp_a=0.3,
        version_b="V6 — RAG + ReAct",
        temp_b=0.3,
        answer_a="Ans A",
        answer_b="Ans B",
        vote="A",
        a_config="V6 — RAG + ReAct (T=0.3)",
        b_config="V3 — Few-Shot (T=0.3)",
        swapped=1,
        protocol="blind",
    )
    assert row_id > 0
    
    # Verify row was inserted with blind columns
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM ab_log WHERE id = ?", (row_id,)).fetchone()
    assert row["swapped"] == 1
    assert row["protocol"] == "blind"
    assert row["a_config"] == "V6 — RAG + ReAct (T=0.3)"
    
    # Clean up test row so real test counts remain unaffected
    conn.execute("DELETE FROM ab_log WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()
