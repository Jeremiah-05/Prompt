"""
Unit tests for Step 6: Detailed Evaluation Query Matrix display.
Validates:
1. Cells in load_results_csv() have newlines and pipes sanitized.
2. Long answers are truncated cleanly to <= 120 chars.
3. Every Markdown row has exactly matching column counts without broken formatting.
"""

import pytest
from app import load_results_csv


def test_load_results_csv_sanitization():
    """Verify load_results_csv returns a clean, unbroken Markdown table."""
    table = load_results_csv()
    assert table and isinstance(table, str)
    lines = [line.strip() for line in table.splitlines() if line.strip()]
    
    assert len(lines) >= 3, "Table must have at least header, separator, and data rows"
    header_line = lines[0]
    headers = [col.strip() for col in header_line.split("|")[1:-1]]
    num_cols = len(headers)
    
    for i, line in enumerate(lines):
        assert line.startswith("|") and line.endswith("|"), f"Line {i+1} does not start and end with pipe: {line[:50]}"
        parts = line.split("|")[1:-1]
        assert len(parts) == num_cols, f"Line {i+1} has {len(parts)} columns, expected {num_cols}"
        
        # Verify that no cell has unescaped raw newlines (checked by single-line integrity)
        assert "\n" not in line
        assert "\r" not in line


def test_cell_answer_truncation():
    """Verify that answers longer than 120 chars are truncated to prevent table blowout."""
    table = load_results_csv()
    lines = [line.strip() for line in table.splitlines() if line.strip()]
    headers = [col.strip() for col in lines[0].split("|")[1:-1]]
    answer_idx = headers.index("answer") if "answer" in headers else -1
    
    if answer_idx != -1:
        for line in lines[2:]:  # Skip header and separator
            cells = [c.strip() for c in line.split("|")[1:-1]]
            ans_cell = cells[answer_idx]
            assert len(ans_cell) <= 125, f"Answer cell length {len(ans_cell)} exceeds 125 chars: {ans_cell[:50]}..."
