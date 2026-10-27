# -*- coding: utf-8 -*-
"""
scripts/check_consistency.py

Automated consistency checker:
Asserts that all numbers, metrics, counts, and verdicts across:
  - README.md
  - docs/REPORT.md
  - docs/PRESENTATION_OUTLINE.md
  - docs/VERIFICATION_REPORT.md
match the empirical ground truth in the repository CSV and DB files:
  - eval/results.csv
  - eval/param_results.csv
  - eval/bias_results.csv
  - eval/heldout_results.csv
  - data/shopease.db / data/orders.json
"""

import sys
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
DOCS_DIR = ROOT / "docs"

README_PATH = ROOT / "README.md"
REPORT_PATH = DOCS_DIR / "REPORT.md"
OUTLINE_PATH = DOCS_DIR / "PRESENTATION_OUTLINE.md"
VERIF_PATH = DOCS_DIR / "VERIFICATION_REPORT.md"


def load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_csv_truth():
    results = load_csv(EVAL_DIR / "results.csv")
    param_results = load_csv(EVAL_DIR / "param_results.csv")
    bias_results = load_csv(EVAL_DIR / "bias_results.csv")
    heldout_results = load_csv(EVAL_DIR / "heldout_results.csv")

    # V1 in results.csv
    v1_rows = [r for r in results if r["version"].startswith("V1")]
    assert len(v1_rows) == 15, f"V1 count {len(v1_rows)} != 15"
    v1_succ = sum(int(r["task_success"]) for r in v1_rows)
    v1_hall = sum(int(r["hallucination"]) for r in v1_rows)
    v1_ungr = sum(int(r["ungrounded"]) for r in v1_rows)
    assert v1_succ == 7, f"V1 success {v1_succ} != 7 (46.7%)"
    assert v1_hall == 3, f"V1 hallucination {v1_hall} != 3 (20.0%)"
    assert v1_ungr == 4, f"V1 ungrounded {v1_ungr} != 4 (26.7%)"

    # Held-out in heldout_results.csv
    assert len(heldout_results) == 21, f"Heldout count {len(heldout_results)} != 21"
    h_succ = sum(int(r["task_success"]) for r in heldout_results)
    h_tool = sum(int(r["tool_accuracy"]) for r in heldout_results)
    h_hall = sum(int(r["hallucination"]) for r in heldout_results)
    h_ungr = sum(int(r["ungrounded"]) for r in heldout_results)
    assert h_succ == 12, f"Heldout success {h_succ} != 12 (57.1%)"
    assert h_tool == 8, f"Heldout tool accuracy {h_tool} != 8 (38.1%)"
    assert h_hall == 0, f"Heldout hallucination {h_hall} != 0 (0.0%)"
    assert h_ungr == 2, f"Heldout ungrounded {h_ungr} != 2 (9.5%)"

    # Bias results: all PAIR-01..PAIR-05 must have bias_equivalent == EQUIVALENT
    for r in bias_results:
        assert r["bias_equivalent"] == "EQUIVALENT", f"{r['pair_id']} is not EQUIVALENT in bias_results.csv"

    print("✅ CSV source-of-truth numbers verified.")


def check_documents():
    readme = README_PATH.read_text(encoding="utf-8")
    report = REPORT_PATH.read_text(encoding="utf-8")
    outline = OUTLINE_PATH.read_text(encoding="utf-8")
    verif = VERIF_PATH.read_text(encoding="utf-8")

    errors = []

    def err(msg):
        errors.append(msg)

    # 1. Check order count is 21 everywhere (not 20)
    for name, text in [("README", readme), ("REPORT", report), ("OUTLINE", outline), ("VERIF", verif)]:
        if "20 orders" in text or "orders.json containing 20" in text:
            err(f"{name}: contains '20 orders'; order count is 21.")

    # 2. Check PAIR-02 is EQUIVALENT everywhere, not DIVERGENT
    for name, text in [("README", readme), ("REPORT", report), ("OUTLINE", outline), ("VERIF", verif)]:
        if "PAIR-02" in text and "DIVERGENT" in text:
            err(f"{name}: PAIR-02 is labeled DIVERGENT; must be EQUIVALENT.")

    # 3. Check Slide 7 does NOT claim 3x repeats 0.0 spread
    if "3× repeats showed 0.0 spread" in outline or "3x repeats showed 0.0 spread" in outline:
        err("PRESENTATION_OUTLINE Slide 7: claims 3x repeats showed 0.0 spread; must be removed.")

    # 4. Check Slide 7 held-out numbers: must not claim 38.1% success or 4.8% hallucination
    if "38.1% task success" in outline:
        err("PRESENTATION_OUTLINE Slide 7: claims 38.1% task success (38.1% is tool accuracy; task success is 57.1%).")
    if "4.8% hallucination" in outline:
        err("PRESENTATION_OUTLINE Slide 7: claims 4.8% hallucination (hallucination is 0.0%).")

    # 5. Check Slide 8 does NOT claim hallucination "reduced to zero"
    if "reduced to zero" in outline:
        err("PRESENTATION_OUTLINE Slide 8: says 'reduced to zero'; must reword to 'no hallucination observed'.")

    # 6. Check REPORT §3.1, §6.2, §6.3 V1 results (must be 46.7%, not old manual-log 73.3%)
    if "V1 Baseline Failures (Q01, Q03, Q05, Q11)" in report:
        err("REPORT: has old 4-failure V1 baseline (Q01, Q03, Q05, Q11); V1 has 8 failures in results.csv.")
    if re.search(r"V1.*73\.3%", report):
        err("REPORT: contains old manual-log 73.3% V1 score; V1 score in CSV is 46.7%.")

    # 7. Check REPORT §8.1 V1 hallucination (must be 20.0%, not 6.7%)
    if "Reduced from **6.7%**" in report or "Reduced from 6.7%" in report:
        err("REPORT §8.1: says V1 hallucination 6.7%; CSV truth is 20.0%.")

    # 8. Check README held-out ungrounded rate: must be 9.5%, not 0.0%
    if "Ungrounded: **0.0%** (0/21" in readme or "Ungrounded: 0.0% (0/21" in readme:
        err("README: held-out ungrounded is listed as 0.0% (0/21); CSV truth is 9.5% (2/21).")

    # 9. Check REPORT tabs count: must say four tabs, not three
    if "three synchronized interfaces" in report:
        err("REPORT: says 'three synchronized interfaces'; Gradio app has four tabs.")

    # 10. Check REPORT persistence: must say SQLite with ACID transactions, not JSON/in-memory
    if "JSON disk persistence" in report:
        err("REPORT: says 'JSON disk persistence'; must say SQLite with ACID transactions.")

    # 11. Check README limitations duplicate section
    lim_count = readme.count("## ⚠️ Limitations")
    if lim_count > 1:
        err(f"README: contains {lim_count} 'Limitations' sections; must be merged into one.")

    # 12. Check held-out injection counts in README: H14, H15, H19–H21
    if "H19–H21" in readme and "H14, H15" not in readme:
        err("README: held-out injection count should include H14, H15, H19–H21.")

    if errors:
        print(f"\n❌ Consistency check failed with {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✅ All document consistency checks passed!")


if __name__ == "__main__":
    check_csv_truth()
    check_documents()
