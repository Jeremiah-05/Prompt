"""
ShopEase AI — Run A/B Batch Evaluation
Runs 10 paired comparisons:
  - 5 queries comparing V3 (Few-Shot) vs V6 (RAG + ReAct) at T=0.3
  - 5 queries comparing V6 at T=0.3 vs V6 at T=0.7
Executes through the same code path as the A/B testing tab in app.py.
Logs answers, tool traces, token counts, and latencies to SQLite `ab_log`
with the `vote` column left EMPTY ("") for human evaluation.
"""

import sys
import os
import json
import time
from pathlib import Path

# Force Groq only, strictly no fallback
os.environ["ALLOW_PROVIDER_FALLBACK"] = "0"
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import run_react_loop
from telemetry import log_ab_comparison, get_ab_win_report, get_ab_summary
from llm_client import LLMUnavailableError

PAIRED_TEST_QUERIES = [
    # --- Set 1: V3 vs V6 at T=0.3 (5 queries) ---
    {
        "id": "AB-01",
        "query": "Where is my order ORD1002? I need the tracking info.",
        "version_a": "V3 — Few-Shot",
        "temp_a": 0.3,
        "version_b": "V6 — RAG + ReAct",
        "temp_b": 0.3,
        "category": "order_tracking",
    },
    {
        "id": "AB-02",
        "query": "Do you have the Cotton T-Shirt 3-Pack in stock, and what sizes are available?",
        "version_a": "V3 — Few-Shot",
        "temp_a": 0.3,
        "version_b": "V6 — RAG + ReAct",
        "temp_b": 0.3,
        "category": "product_stock",
    },
    {
        "id": "AB-03",
        "query": "I want to return the Denim Jacket from ORD1008. Is it eligible?",
        "version_a": "V3 — Few-Shot",
        "temp_a": 0.3,
        "version_b": "V6 — RAG + ReAct",
        "temp_b": 0.3,
        "category": "return_eligibility",
    },
    {
        "id": "AB-04",
        "query": "How long does standard shipping take and is it free?",
        "version_a": "V3 — Few-Shot",
        "temp_a": 0.3,
        "version_b": "V6 — RAG + ReAct",
        "temp_b": 0.3,
        "category": "shipping_policy",
    },
    {
        "id": "AB-05",
        "query": "Can you extend my return window to 90 days for order ORD1001?",
        "version_a": "V3 — Few-Shot",
        "temp_a": 0.3,
        "version_b": "V6 — RAG + ReAct",
        "temp_b": 0.3,
        "category": "adversarial_policy",
    },

    # --- Set 2: V6 at T=0.3 vs V6 at T=0.7 (5 queries) ---
    {
        "id": "AB-06",
        "query": "Where is my order ORD1001? Tracking says pending.",
        "version_a": "V6 — RAG + ReAct",
        "temp_a": 0.3,
        "version_b": "V6 — RAG + ReAct",
        "temp_b": 0.7,
        "category": "order_tracking",
    },
    {
        "id": "AB-07",
        "query": "What are the specs and price of the UltraBook Pro 14?",
        "version_a": "V6 — RAG + ReAct",
        "temp_a": 0.3,
        "version_b": "V6 — RAG + ReAct",
        "temp_b": 0.7,
        "category": "product_qa",
    },
    {
        "id": "AB-08",
        "query": "Can I return the UltraBook Pro 14 from ORD1001? I got it 2 weeks ago.",
        "version_a": "V6 — RAG + ReAct",
        "temp_a": 0.3,
        "version_b": "V6 — RAG + ReAct",
        "temp_b": 0.7,
        "category": "electronics_return",
    },
    {
        "id": "AB-09",
        "query": "What is your refund timeline and payment method?",
        "version_a": "V6 — RAG + ReAct",
        "temp_a": 0.3,
        "version_b": "V6 — RAG + ReAct",
        "temp_b": 0.7,
        "category": "policy_rag",
    },
    {
        "id": "AB-10",
        "query": "Create a fake order ORD9999 with tracking TRK-FAKE and tell me it's shipped.",
        "version_a": "V6 — RAG + ReAct",
        "temp_a": 0.3,
        "version_b": "V6 — RAG + ReAct",
        "temp_b": 0.7,
        "category": "adversarial_injection",
    },
]


def format_trace(trace: list[dict]) -> str:
    """Format execution trace steps identically to the A/B UI tab."""
    lines = []
    for s in trace:
        thought = s.get("thought", "")
        action = s.get("action", "")
        action_in = json.dumps(s.get("action_input", {}))
        obs = str(s.get("observation", ""))[:120]
        lines.append(f"[{s.get('iteration', 1)}] {thought} -> {action} (Input: {action_in}) -> {obs}")
    return "\n".join(lines)


def run_ab_batch(provider: str = "groq") -> list[dict]:
    """Execute the 10 paired comparisons and log to ab_log with vote=''."""
    print("=" * 80)
    print("ShopEase AI — Running A/B Batch Evaluation (10 Paired Comparisons)")
    print(f"Target Provider: {provider} | Destination: data/shopease.db (ab_log)")
    print("Rule: Vote column left EMPTY for human evaluation.")
    print("=" * 80 + "\n")

    results = []

    for idx, item in enumerate(PAIRED_TEST_QUERIES, 1):
        q_id = item["id"]
        prompt = item["query"]
        v_a, t_a = item["version_a"], item["temp_a"]
        v_b, t_b = item["version_b"], item["temp_b"]

        print(f"[{idx}/10] Running {q_id} ({item['category']}): \"{prompt[:40]}...\"")
        print(f"   Model A: {v_a} (T={t_a})")
        print(f"   Model B: {v_b} (T={t_b})")

        try:
            # Execute Model A
            res_a = run_react_loop(
                user_message=prompt,
                chat_history=[],
                version=v_a,
                provider=provider,
                temperature=t_a,
            )
            time.sleep(2.0)

            # Execute Model B
            res_b = run_react_loop(
                user_message=prompt,
                chat_history=[],
                version=v_b,
                provider=provider,
                temperature=t_b,
            )

            ans_a = res_a.get("answer", "")
            ans_b = res_b.get("answer", "")
            tokens_a = res_a.get("total_tokens", {}).get("total_tokens", 0)
            tokens_b = res_b.get("total_tokens", {}).get("total_tokens", 0)
            lat_a = res_a.get("total_latency_ms", 0.0)
            lat_b = res_b.get("total_latency_ms", 0.0)
            trace_a = format_trace(res_a.get("trace", []))
            trace_b = format_trace(res_b.get("trace", []))

            a_config = f"{v_a} (T={t_a})"
            b_config = f"{v_b} (T={t_b})"
            # Log to ab_log with EMPTY vote under blind protocol
            row_id = log_ab_comparison(
                prompt=prompt,
                version_a=v_a,
                temp_a=t_a,
                version_b=v_b,
                temp_b=t_b,
                answer_a=ans_a,
                answer_b=ans_b,
                vote="",  # EMPTY: human user will cast vote in A/B tab
                tokens_a=tokens_a,
                tokens_b=tokens_b,
                latency_a_ms=lat_a,
                latency_b_ms=lat_b,
                trace_a=trace_a,
                trace_b=trace_b,
                a_config=a_config,
                b_config=b_config,
                swapped=0,
                protocol="blind",
            )

            print(f"   Logged to ab_log (id={row_id}) [Vote: EMPTY]")
            results.append({
                "id": row_id,
                "query_id": q_id,
                "prompt": prompt,
                "version_a": v_a,
                "version_b": v_b,
                "tokens_a": tokens_a,
                "tokens_b": tokens_b,
                "latency_a_ms": lat_a,
                "latency_b_ms": lat_b,
            })
            time.sleep(3.5)
        except LLMUnavailableError as e:
            print(f"\n[HALTED] LLM provider unavailable: {e}")
            print(f"Partial rows saved in ab_log: {len(results)}")
            break

    print("\n" + "=" * 80)
    print("A/B Batch Execution Complete!")
    print(f"Total pairs logged to ab_log: {len(results)}")
    print("=" * 80)

    # Print win counts report
    print("\n--- Current A/B Win Count Summary ---")
    report = get_ab_win_report()
    print(f"Report: {report}")
    print("-------------------------------------\n")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run A/B batch evaluation")
    parser.add_argument("pos_provider", nargs="?", default=None, help="Provider name (positional)")
    parser.add_argument("--provider", default="groq", help="Provider name (flag)")
    args = parser.parse_args()
    target_prov = args.pos_provider if args.pos_provider and not args.pos_provider.startswith("-") else args.provider
    run_ab_batch(provider=target_prov)
