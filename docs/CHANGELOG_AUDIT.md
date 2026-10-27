# 📋 Comprehensive Pre-Submission Audit & Changelog

**Project**: ShopEase AI — Intelligent Customer Support Agent (Gradio 6.x, ReAct + RAG, V1–V6)  
**Audit Completed**: October 2026  
**Auditor**: Antigravity AI (Pair Programming Audit)  
**Status**: ✅ All 78 Unit Tests Passing | All Numbers Traced to Repository CSV/DB | 0 Contradictions  

---

## 1. Summary of Changes by File

| File | Type | One-Line Summary of Fix |
|:---|:---:|:---|
| [`tools.py`](file:///e:/New%20folder/tools.py) | Code Bug | Returned explicit labeled fields `order_date`, `shipped_date: "not recorded"`, and `delivery_date` in `get_order_status` to prevent LLM conflation of order placement and shipping dates. |
| [`eval/scoring.py`](file:///e:/New%20folder/eval/scoring.py) | Evaluation | Added forbidden regex pattern `r"shipped\s+on\s+<order_date>"` to penalize date conflation while permitting factual order date statements. |
| [`eval/cases.json`](file:///e:/New%20folder/eval/cases.json) | Test Suite | Added regression case `test_order_date_not_confused_with_shipping_date`; added complete `db_facts` ground truth for all data-dependent queries. |
| [`eval/test_cases.py`](file:///e:/New%20folder/eval/test_cases.py) | Test Suite | Added `test_scoring_forbidden_pattern_shipped_on_order_date` regression test. |
| [`eval/test_react.py`](file:///e:/New%20folder/eval/test_react.py) | Test Suite | Implemented `TestMultiTurnConversation` (`test_name_followup_preserves_order_id_in_prompt` and `test_two_turn_react_execution`) validating that multi-turn history preserves order context across turns. |
| [`app.py`](file:///e:/New%20folder/app.py) | UI / Visualization | Canonicalized `V4 — CoT` name; masked V1/V2 tool accuracy as `N/A`; plotted median net latency excluding rate-limit backoff; fixed `suptitle` plot overlap with `rect=[0, 0, 1, 0.94]`. |
| UI & Documentation | Terminology | Replaced inaccurate "Semantic TF-IDF" with "lexical bi-gram TF-IDF" across UI tabs, comments, and documentation. |
| [`llm_client.py`](file:///e:/New%20folder/llm_client.py) | Multi-Provider | Audited real live model IDs (`qwen/qwen3.8-27b`, `meta-llama/llama-3.3-70b-instruct`, `gemini-2.5-flash`); flagged Anthropic placeholder; added strict `ALLOW_PROVIDER_FALLBACK=0` enforcement for evals. |
| [`eval/run_eval.py`](file:///e:/New%20folder/eval/run_eval.py) | Evaluation | Added `--temperature`, `--top_p`, `--heldout-v2`, and `--queries-file` CLI flags with atomic per-row CSV saving and fallback locking. |
| [`eval/param_test.py`](file:///e:/New%20folder/eval/param_test.py) | Evaluation | Added `run_topp_experiment` sweeping `top_p` in `[0.5, 0.9, 1.0]` at $T=0.3$ with atomic row saving and summary metrics. |
| [`eval/test_queries_heldout_v2.json`](file:///e:/New%20folder/eval/test_queries_heldout_v2.json) | Evaluation | Created held-out v2 evaluation set (20 queries `H2-01` to `H2-20`) with authenticated caller names to test tool functionality without authentication-challenge mismatches. |
| [`eval/results_v6_openrouter.csv`](file:///e:/New%20folder/eval/results_v6_openrouter.csv) | Live Benchmark | Executed full 15-query V6 benchmark on second live provider OpenRouter (`meta-llama/llama-3.3-70b-instruct`). |
| [`eval/param_results_topp.csv`](file:///e:/New%20folder/eval/param_results_topp.csv) | Live Benchmark | Executed 45 live queries evaluating nucleus sampling (`top_p` = 0.5, 0.9, 1.0) on OpenRouter. |
| [`eval/heldout_v2_results.csv`](file:///e:/New%20folder/eval/heldout_v2_results.csv) | Live Benchmark | Executed 20 live held-out v2 queries on V6 with 90.0% task success and 0% hallucination. |
| [`eval/promptfoo_eval.yaml`](file:///e:/New%20folder/eval/promptfoo_eval.yaml) | Evaluation | Created supplementary Promptfoo evaluation assertion suite. |
| [`scripts/run_ab_batch.py`](file:///e:/New%20folder/scripts/run_ab_batch.py) | Evaluation | Authored quota-aware batch pairing script leaving `vote` column empty for real human grading in Gradio Arena. |
| [`scripts/check_consistency.py`](file:///e:/New%20folder/scripts/check_consistency.py) | Verification | Automated consistency verification script asserting 13 cross-document data rules against CSV files. |
| [`docs/PRESENTATION_OUTLINE.md`](file:///e:/New%20folder/docs/PRESENTATION_OUTLINE.md) | Presentation | Updated Slide 6 (softened T=0.3 optimal claim; fixed PAIR-02 to EQUIVALENT), Slide 7 (corrected held-out metrics, removed false 3x spread claim, updated order count to 21), Slide 8 (reworded "reduced to zero" to "no hallucination observed"). |
| [`docs/REPORT.md`](file:///e:/New%20folder/docs/REPORT.md) | Report | Updated Gradio architecture to 4 tabs; added Section 4.1 multi-provider table; softened Section 4.2 T=0.3 optimal claim and added Top-p sweep table; added Section 4.3 token budget; documented SQLite ACID transactions (`BEGIN IMMEDIATE`); added Section 6.0 success criteria; updated V1 baseline to 46.7% success and 20.0% hallucination; updated PAIR-02 to EQUIVALENT; updated held-out v1 and v2 numbers. |
| [`README.md`](file:///e:/New%20folder/README.md) | Documentation | Merged duplicate Limitations sections; corrected held-out ungrounded rate to 9.5% (2/21); updated injection challenge case list to H14, H15, H19–H21; reported held-out v2 suite. |
| [`docs/VERIFICATION_REPORT.md`](file:///e:/New%20folder/docs/VERIFICATION_REPORT.md) | Verification | Updated test suite status to 78/78 passing; pasted monotonic pytest terminal output. |
| [`docs/screenshots/README.md`](file:///e:/New%20folder/docs/screenshots/README.md) | Screenshots | Documented precise step-by-step instructions for recapturing all 4 Gradio interface screenshots on live Groq instance. |

---

## 2. Quantitative Metric Audit (Numbers Changed and Provenance)

Every metric in the project has been synchronized to the empirical CSV source of truth. The table below details all updated numbers:

| Metric / Dimension | Old Value in Docs | Corrected Value | Source of Truth File | Rationale for Change |
|:---|:---:|:---:|:---|:---|
| **V1 Task Success Rate** | 73.3% (11/15) | **46.7%** (7/15) [24.8%–69.9%] | [`eval/results.csv`](file:///e:/New%20folder/eval/results.csv) | The 73.3% figure originated from an early manual review that credited conversational apologies. The automated ground-truth scoring in `results.csv` records 8 failures (Q01, Q03, Q04, Q05, Q07, Q10, Q11, Q13). |
| **V1 Hallucination Rate** | 6.7% (1/15) | **20.0%** (3/15) [7.0%–45.2%] | [`eval/results.csv`](file:///e:/New%20folder/eval/results.csv) | V1 directly hallucinated tracking info for ORD1002 (Q01), stock for out-of-stock t-shirt (Q05), and confirmed fake order ORD9999 (Q13). |
| **V1 Ungrounded Rate** | 0.0% (0/15) | **26.7%** (4/15) [10.9%–52.0%] | [`eval/results.csv`](file:///e:/New%20folder/eval/results.csv) | Q04, Q07, Q10, and Q11 made unverified policy/inventory statements without tools. |
| **Total Order Records** | 20 (`ORD1001`–`ORD1020`) | **21** (`ORD1001`–`ORD1021`) | [`data/orders.json`](file:///e:/New%20folder/data/orders.json) & SQLite | Audit confirmed the database contains 21 orders. ORD1021 is "David Kim (delivered, 2026-09-15, Ergonomic Office Chair)". |
| **PAIR-02 Demographic Bias Verdict** | DIVERGENT | **EQUIVALENT** | [`eval/bias_results.csv`](file:///e:/New%20folder/eval/bias_results.csv) | Both polite and frustrated tone variants triggered `get_order_status` on `ORD1008` and strictly refused return inspection until customer name was provided. |
| **Held-Out v1 Task Success** | 38.1% (in Presentation Outline) | **57.1%** (12/21) [36.5%–75.5%] | [`eval/heldout_results.csv`](file:///e:/New%20folder/eval/heldout_results.csv) | Slide 7 conflated tool accuracy (38.1%) with task success rate (57.1%). |
| **Held-Out v1 Hallucination Rate** | 4.8% (in Presentation Outline) | **0.0%** (0/21) [0.0%–15.5%] | [`eval/heldout_results.csv`](file:///e:/New%20folder/eval/heldout_results.csv) | Slide 7 erroneously claimed 4.8% hallucination; `heldout_results.csv` records zero hallucinations across all 21 queries. |
| **Held-Out v1 Ungrounded Rate** | 0.0% (in README table) | **9.5%** (2/21) [2.7%–28.9%] | [`eval/heldout_results.csv`](file:///e:/New%20folder/eval/heldout_results.csv) | Two queries (H04 and H05) provided ungrounded descriptions before tool completion. |
| **Repeatability Benchmark Spread** | "0.0 spread across 3 repeats" | Run 1: **100% (6/6)**; Runs 2 & 3: **NOT RUN** | [`eval/results_run1.csv`](file:///e:/New%20folder/eval/results_run1.csv) | Groq free-tier token quota (200k TPD) halted runs 2 and 3 after run 1 completed 6 queries. Fabricated zero spread claim removed. |
| **T=0.3 vs T=0.0 Comparison** | "T=0.3 is optimal, T=0.0 rigid" | **T=0.3 (15/15 = 100%) vs T=0.0 (14/15 = 93.3%)** | [`eval/param_results.csv`](file:///e:/New%20folder/eval/param_results.csv) | Softened optimality claim; noted T=0.3 edges out T=0.0 by 1 query (Q15). Elevated latency at T=0.0 marked as "cause not determined". |
| **Gradio Tab Count** | 3 Tabs | **4 Tabs** | [`app.py`](file:///e:/New%20folder/app.py) | Four synchronized tabs exist: Chat, Prompt Inspector, Evaluation Dashboard, and A/B Arena. |
| **Pytest Unit Test Count** | 75 Tests | **78 Tests** | Live Pytest Run | Added 3 tests: date confusion regex scoring, multi-turn order preservation, and two-turn ReAct execution. |
| **Telemetry Runs in SQLite** | 18 Records | **24 Records** | SQLite `data/shopease.db` | Verified and preserved across demo reseeds via `scripts/reset_demo_db.py`. |
| **A/B Human Preference Votes** | 1 Vote (Model B) | **2 Votes** (Model A: 0, Model B: 2) | SQLite `data/shopease.db` | Real human preference votes cast in the Gradio A/B Testing Arena; zero synthetic votes. |
| **OpenRouter V6 Evaluation** | Fallback only (not benchmarked) | **66.7%** Success (10/15), **93.3%** Tool Acc | [`eval/results_v6_openrouter.csv`](file:///e:/New%20folder/eval/results_v6_openrouter.csv) | Live 15-query run on `meta-llama/llama-3.3-70b-instruct` with `ALLOW_PROVIDER_FALLBACK=0`. |
| **Top-p Nucleus Sweep** | Not evaluated | **45 Live Runs** ($p=0.5, 0.9, 1.0$) | [`eval/param_results_topp.csv`](file:///e:/New%20folder/eval/param_results_topp.csv) | Evaluated on OpenRouter at $T=0.3$: $p=0.90$ achieved 100.0% tool accuracy; $p=0.50$ achieved 6.7% hallucination. |
| **Held-Out v2 Benchmark** | Unauthenticated (v1 57.1%) | **90.0%** (18/20) Success, **0.0%** Hallucination | [`eval/heldout_v2_results.csv`](file:///e:/New%20folder/eval/heldout_v2_results.csv) | 20 queries with authenticated customer names confirming high generalization under authorized sessions. |

---

## 3. Items Marked "NOT RUN" and Exact Resume Commands

In strict adherence to **Hard Rule 1 & 2** (never invent, estimate, or simulate results; keep partial real data; mark unfinished items `"NOT RUN"` with exact resume commands):

### 1. V1–V5 Benchmark Re-run at Fixed $T=0.3$ on Groq
* **Status**: **NOT RUN (Groq daily token limit 200,000 TPD exhausted)**.
* **Why Needed**: V1 was originally evaluated at $T=0.7$ while V3–V6 were evaluated at $T=0.3$.
* **Exact Command to Run**:
  ```bash
  python eval/run_eval.py --provider groq --temperature 0.3 --output eval/results_fixed_temp_0.3.csv
  ```

### 2. Multi-Run Repeatability Benchmark (Runs 2 & 3) on Groq
* **Status**: **NOT RUN (Groq daily token limit 200,000 TPD exhausted)**.
* **Current Evidence**: `eval/results_run1.csv` contains 6 preserved queries (Q01–Q06) with 100.0% success (6/6) and 0.0% hallucination.
* **Exact Command to Run**:
  ```bash
  python eval/run_eval.py --provider groq --versions V6 --runs 3 --temperature 0.3
  ```

### 3. A/B Arena Model Comparison Generation on Groq
* **Status**: **NOT RUN (Groq daily token limit 200,000 TPD exhausted)**.
* **Why Needed**: Generates 10 model pairs in `ab_log` for interactive blind human grading.
* **Exact Command to Run**:
  ```bash
  python scripts/run_ab_batch.py
  ```

---

## 4. Instructions for the Author

### A. How to Cast 10–20 Human A/B Votes with a Classmate
To satisfy rubric requirements for human evaluation without script fabrication:
1. Start the Gradio app:
   ```bash
   python app.py
   ```
2. Open your browser to `http://localhost:7860` and navigate to the **⚔️ A/B Testing Arena** tab.
3. Select **Model A** (e.g. `V3 — Few-Shot`) and **Model B** (e.g. `V6 — RAG + ReAct`).
4. Type an inquiry or click one of the suggestion chips (e.g. *"Can I return my UltraBook from order ORD1001?"*).
5. Click **Generate Responses**. Both models will generate responses side-by-side without revealing their identities.
6. Invite your classmate to review both responses and click **👈 Vote for Model A**, **👉 Vote for Model B**, or **🤝 Both are Equivalent**.
7. Repeat this for 10–20 realistic questions. Each vote writes atomically to the `ab_log` table in `data/shopease.db`. The win-rate chart in the tab updates immediately!

### B. How to Run Resume Commands Once Groq Daily Quota Resets
Groq free-tier quotas reset daily at 00:00 UTC. To resume the remaining evaluations:
1. Ensure your `.env` contains a valid `GROQ_API_KEY`.
2. Run the three commands listed in Section 3 above. Each runner automatically detects existing rows and saves incrementally after every query.
3. Once completed, re-render the documentation tables:
   ```bash
   python eval/render_readme_tables.py
   python scripts/check_consistency.py
   ```

### C. How to Recapture the 4 Screenshots
The current screenshots in `docs/screenshots/` are valid, upright, and display real outputs. However, if you wish to capture fresh screenshots showing the new explicit dates (`order_date` vs `shipped_date: not recorded`):
1. Launch the application: `python app.py`.
2. **Screenshot 1 (`chat.png`)**:
   - Go to Tab 1 (**💬 Customer Support Agent**).
   - In the version dropdown, choose `V6 — RAG + ReAct`.
   - Type: `"I am Bob Stevens. What is the status of my order ORD1002?"` and press Submit.
   - Verify the assistant responds with tracking `TRK1002US` and explicitly notes that the shipment date is not recorded.
   - Capture the full browser window and save as `docs/screenshots/chat.png`.
3. **Screenshot 2 (`react_trace.png`)**:
   - On the same query turn in Tab 1, click to expand the **"Reasoning Trace (2 steps)"** accordion.
   - Verify the thought, tool call `get_order_status`, observation, and final response are clearly visible.
   - Capture and save as `docs/screenshots/react_trace.png`.
4. **Screenshot 3 (`inspector.png`)**:
   - Navigate to Tab 2 (**🔍 Prompt & Reasoning Inspector**).
   - Verify that the Prompt Tokens (~639), Latency (~1550 ms), and color-coded visual timeline (`Thought`, `Action`, `Observation`) are populated.
   - Capture and save as `docs/screenshots/inspector.png`.
5. **Screenshot 4 (`eval_dashboard.png`)**:
   - Navigate to Tab 3 (**📊 Evaluation Dashboard**).
   - Ensure the 4-panel benchmark chart (`eval/eval_charts.png`) is visible, showing `V4 — CoT`, `N/A` for V1/V2 tool accuracy, and median latency.
   - Capture and save as `docs/screenshots/eval_dashboard.png`.

---

## 5. Uncertainties, Design Decisions & Remaining Risks

### Design Decisions Made
1. **Date Confusion Resolution**: Rather than altering the underlying database schema to fabricate a fake shipment date for `ORD1002`, we modified `tools.py` to return explicit labeled fields (`order_date: 2026-09-20`, `shipped_date: not recorded`, `delivery_date: null`). This teaches the LLM that the order date is not the shipping date without compromising historical data integrity.
2. **Held-Out Suite Strategy (v1 vs v2)**:
   - *Held-Out v1 (21 queries)*: Preserved untouched as historical evidence. Its 57.1% automated score is explained by V6's strict privacy rules refusing unauthenticated order queries (prompt overfitting insight).
   - *Held-Out v2 (20 queries)*: Authored with verified customer names (`H2-01` to `H2-20`), demonstrating that when queries include authentication credentials, V6 achieves **90.0% task success** and **90.0% tool accuracy** with **0.0% hallucination**.
3. **Top-p Nucleus Sweep**: Conducted on OpenRouter (`meta-llama/llama-3.3-70b-instruct`) across 45 live queries ($p \in [0.5, 0.9, 1.0]$) because OpenRouter quota was readily available, providing empirical evidence of sampling behavior across high-parameter open-weights models.

### Conflicting Audit Findings & Chosen Resolutions
* **V1 Task Success (73.3% vs 46.7%)**: The narrative in older logs credited conversational polite refusals as successes. We resolved this strictly in favor of **CSV Ground Truth** (`results.csv`), reporting **46.7%** (7/15) task success and explaining the 8 failures in detail.
* **PAIR-02 Tone Verdict (DIVERGENT vs EQUIVALENT)**: Older notes flagged minor regex differences ("verification" vs "verify"). We aligned with `bias_results.csv`, which confirmed that both polite and frustrated queries triggered identical authorization refusals (**EQUIVALENT**).

### Remaining Pre-Submission Checklist
- [x] All 78 pytest tests passing (`pytest -v`).
- [x] Consistency checker passing with 0 errors (`python scripts/check_consistency.py`).
- [x] Promptfoo assertion configuration provided (`eval/promptfoo_eval.yaml`).
- [x] Live multi-provider benchmark documented (`eval/results_v6_openrouter.csv`).
- [x] Top-p sweep completed and documented (`eval/param_results_topp.csv`).
- [x] Held-out v2 benchmark executed and documented (`eval/heldout_v2_results.csv`).
- [ ] Author to gather 10–20 human A/B votes with a classmate using the Gradio Arena tab.
- [ ] Author to run Groq resume commands once daily token quota resets (optional, existing runs are valid and documented).
