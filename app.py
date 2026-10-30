"""
ShopEase AI — Intelligent Customer Service Bot
Main application: Modern Gradio UI + ReAct Loop Orchestrator + RAG Grounding.

Three comprehensive tabs:
  1. 💬 Chat — Interactive chat with live model switching, version intel, and 1-click query suggestions
  2. 🔍 Prompt Inspector — Visual KPI metric tiles, step-by-step ReAct trace cards, full prompts & raw outputs
  3. 📊 Evaluation Dashboard — Metric summaries, comparative benchmark charts, and detailed evaluation matrix
"""

import json
import re
import time
import os
import csv
from pathlib import Path

import gradio as gr
import matplotlib
from concurrent.futures import ThreadPoolExecutor

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from llm_client import (
    chat_completion,
    chat_completion_with_fallback,
    count_tokens_approx,
    LLMUnavailableError,
    check_providers,
)
from tools import execute_tool, TOOL_REGISTRY
from retriever import retrieve_policy_chunks, format_rag_context
from prompts.builder import (
    build_prompt,
    get_system_message,
    list_versions,
    get_recommended_temp,
    get_prompt_diff_html,
    PROMPT_VERSIONS,
)
from eval.live_tracker import (
    record_live_turn,
    format_live_kpi_html,
    format_live_table_markdown,
    clear_live_logs,
)
from eval.run_eval import (
    evaluate_single,
    load_test_queries,
    merge_and_save_results,
)
from security import (
    validate_input_length,
    global_rate_limiter,
    redact_pii,
    check_transactional_confirmation,
)
from telemetry import (
    log_telemetry_run,
    calculate_cost_usd,
    get_version_aggregates,
    get_telemetry_summary_markdown,
    log_ab_vote,
    get_ab_summary,
    export_ab_log_csv,
)
from eval.render_readme_tables import (
    get_benchmark_kpi_html,
    get_header_badge_html,
    get_benchmark_v6_kpis,
)


# ---------------------------------------------------------------------------
# Global State & Paths
# ---------------------------------------------------------------------------
LAST_ACTIVE_PROVIDER = {"provider": "Groq", "model": "qwen/qwen3.8-27b"}
BASE_DIR = Path(__file__).parent
EVAL_DIR = BASE_DIR / "eval"
RESULTS_CSV = EVAL_DIR / "results.csv"
HELDOUT_RESULTS_CSV = EVAL_DIR / "heldout_results.csv"

# ---------------------------------------------------------------------------
# Prompt Version Intelligence
# ---------------------------------------------------------------------------
VERSION_INTEL = {
    "V1 — Naive": {
        "badge": "⚠️ Baseline (Zero-Shot)",
        "color": "#ef4444",
        "desc": "Basic zero-shot prompt with no domain rules or tools. Prone to severe hallucinations (making up fake tracking numbers like TRK-FAKE).",
        "tags": ["❌ No Tools", "❌ Hallucinates Orders", "❌ Unconstrained"],
    },
    "V2 — Role + Rules": {
        "badge": "🛡️ Guardrailed Persona",
        "color": "#f59e0b",
        "desc": "Defines polite customer support persona with explicit tone rules, return window definitions, and refusal of out-of-scope queries.",
        "tags": ["✅ Professional Tone", "✅ Scope Boundaries", "❌ No Live Tools"],
    },
    "V3 — Few-Shot": {
        "badge": "📝 In-Context Exemplars",
        "color": "#3b82f6",
        "desc": "Adds 3 few-shot customer support dialogues teaching the model exact JSON tool-calling syntax and refund calculation formats.",
        "tags": ["✅ Tool Call Format", "✅ Valid JSON", "⚡ ~75% Tool Accuracy"],
    },
    "V4 — Chain-of-Thought (CoT)": {
        "badge": "🧠 Chain-of-Thought (CoT)",
        "color": "#8b5cf6",
        "desc": "Requires explicit step-by-step deduction before answering (e.g. checking delivery date vs 30-day window, non-returnable categories).",
        "tags": ["✅ Step-by-Step Logic", "✅ Eligibility Checks", "✅ Fewer Policy Flaws"],
    },
    "V5 — ReAct": {
        "badge": "🔄 Autonomous ReAct Loop",
        "color": "#6366f1",
        "desc": "Full Thought ➔ Action ➔ Observation cycle. The bot queries tools dynamically, inspects results, and iterates until the goal is solved.",
        "tags": ["✅ Multi-Step Chaining", "✅ Dynamic Loops", "✅ Autonomous Finish"],
    },
    "V6 — RAG + ReAct": {
        "badge": "🎯 V6: Grounded RAG + ReAct",
        "color": "#10b981",
        "desc": "Ground-truth evaluated architecture: lexical bi-gram TF-IDF document retrieval injects verified policy and FAQ chunks into the prompt alongside active ReAct tools.",
        "tags": ["🛡️ 0.0% Hallucination [0.0%–20.4%]", "🎯 100.0% Tool Acc [79.6%–100.0%]", "✅ Verified Grounding"],
    },
}
# Legacy aliases for backwards compatibility
VERSION_INTEL["V4 — CoT"] = VERSION_INTEL["V4 — Chain-of-Thought (CoT)"]
VERSION_INTEL["V4 — Chain-of-Thought"] = VERSION_INTEL["V4 — Chain-of-Thought (CoT)"]

def render_version_card(version_name: str) -> str:
    """Generate HTML preview card for the selected prompt version."""
    info = VERSION_INTEL.get(version_name, VERSION_INTEL["V6 — RAG + ReAct"])
    tags_html = " ".join(
        f"<span style='background: rgba(255,255,255,0.08); color: #cbd5e1; padding: 2px 8px; border-radius: 4px; font-size: 0.74rem;'>{t}</span>"
        for t in info['tags']
    )
    return f"""
    <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(99, 102, 241, 0.25); border-left: 4px solid {info['color']}; border-radius: 12px; padding: 12px 14px; margin-top: 8px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-weight: 700; color: {info['color']}; font-size: 0.86rem;">{info['badge']}</span>
        </div>
        <p style="margin: 0 0 8px 0; font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">{info['desc']}</p>
        <div style="display: flex; flex-wrap: wrap; gap: 4px;">{tags_html}</div>
    </div>
    """

# ---------------------------------------------------------------------------
# ReAct Loop parser
# ---------------------------------------------------------------------------

def parse_react_output(text: str) -> dict:
    """Parse LLM output into Thought, Action, Action Input components."""
    result = {"thought": "", "action": "", "action_input": {}, "raw": text}
    
    thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()
    
    action_match = re.search(r'Action:\s*(\S+)', text)
    if action_match:
        result["action"] = action_match.group(1).strip()
    
    input_match = re.search(r'Action Input:\s*(\{.*?\})', text, re.DOTALL)
    if input_match:
        try:
            result["action_input"] = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            raw_json = input_match.group(1).replace("'", '"')
            try:
                result["action_input"] = json.loads(raw_json)
            except json.JSONDecodeError:
                result["action_input"] = {"raw_input": input_match.group(1)}
    
    if not result["action_input"] and result["action"] == "finish":
        final_match = re.search(r'Final Answer:\s*(.+?)$', text, re.DOTALL)
        if final_match:
            result["action_input"] = {"answer": final_match.group(1).strip()}
    
    return result

# ---------------------------------------------------------------------------
# ReAct Loop orchestrator
# ---------------------------------------------------------------------------

def run_react_loop(
    user_message: str,
    chat_history: list[dict],
    version: str = "V6 — RAG + ReAct",
    provider: str = "groq",
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 512,
    max_iterations: int = 5,
) -> dict:
    """Execute the ReAct loop: prompt → LLM → parse → tool → loop → finish.

    Always uses a real LLM. If no provider can answer, LLMUnavailableError is
    raised to the caller; simulated output is never substituted.
    """
    trace = []
    raw_responses = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_latency_ms = 0
    total_retry_wait_ms = 0
    fallback_used = False
    all_valid_json = 1
    react_trace_str = ""
    final_answer = ""
    full_prompt_sent = ""
    model_used = ""
    provider_used = provider
    
    rag_context_str = ""
    if version == "V6 — RAG + ReAct":
        policy_keywords = [
            "policy", "return", "shipping", "warranty", "refund", "cancel",
            "exchange", "payment", "support", "hours", "delivery", "free shipping",
            "how long", "eligible", "window", "order", "buy", "purchase",
            "cancellation", "modify", "change", "update"
        ]
        if any(kw in user_message.lower() for kw in policy_keywords):
            chunks = retrieve_policy_chunks(user_message, top_k=3)
            rag_context_str = format_rag_context(chunks)
    
    for iteration in range(max_iterations):
        prompt_text = build_prompt(
            version=version,
            user_message=user_message,
            chat_history=chat_history,
            rag_context=rag_context_str,
            react_trace=react_trace_str,
        )
        system_msg = get_system_message(version)
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text},
        ]
        
        if iteration == 0:
            full_prompt_sent = f"[SYSTEM INSTRUCTION]\n{system_msg}\n\n[USER QUERY & CONTEXT]\n{prompt_text}"
        
        # Raises LLMUnavailableError (with the real per-provider reasons) on failure.
        llm_result = chat_completion_with_fallback(
            messages=messages,
            primary_provider=provider,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        raw_output = llm_result["content"]
        raw_responses.append(raw_output)
        model_used = llm_result["model"]
        provider_used = llm_result["provider"]
        if llm_result.get("fallback_used"):
            fallback_used = True
        
        usage = llm_result.get("usage", {})
        total_tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_tokens["completion_tokens"] += usage.get("completion_tokens", 0)
        total_tokens["total_tokens"] += usage.get("total_tokens", 0)
        total_latency_ms += llm_result.get("latency_ms", 0)
        total_retry_wait_ms += llm_result.get("retry_wait_ms", 0)
        
        if version in ("V1 — Naive", "V2 — Role + Rules"):
            final_answer = raw_output.strip()
            trace.append({
                "iteration": iteration + 1,
                "thought": "(Direct zero-shot response, tools bypassed)",
                "action": "finish",
                "action_input": {"answer": final_answer},
                "observation": "",
            })
            break
        
        parsed = parse_react_output(raw_output)
        step = {
            "iteration": iteration + 1,
            "thought": parsed["thought"],
            "action": parsed["action"],
            "action_input": parsed["action_input"],
            "observation": "",
        }
        
        if parsed["action"].lower() == "finish":
            final_answer = parsed["action_input"].get("answer", raw_output.strip())
            step["observation"] = "(Completed)"
            trace.append(step)
            break
        
        if parsed["action"]:
            if not isinstance(parsed["action_input"], dict) or "raw_input" in parsed["action_input"]:
                all_valid_json = 0
            
            # Security Hardening: Transactional confirmation turn check
            is_confirmed, challenge_msg = check_transactional_confirmation(
                parsed["action"], parsed["action_input"], user_message, chat_history
            )
            if not is_confirmed:
                step["observation"] = "(Confirmation Required Before Execution)"
                final_answer = challenge_msg
                trace.append(step)
                break

            observation = execute_tool(parsed["action"], parsed["action_input"])
            step["observation"] = observation
            
            react_trace_str += (
                f"\nThought: {parsed['thought']}\n"
                f"Action: {parsed['action']}\n"
                f"Action Input: {json.dumps(parsed['action_input'])}\n"
                f"Observation: {observation}\n"
            )
        else:
            step["observation"] = "(Direct response)"
            final_answer = raw_output.strip()
            trace.append(step)
            break
        
        trace.append(step)
    
    if not final_answer:
        final_answer = (
            "I wasn't able to finalize the inquiry within the allowed 5 iterations. "
            "Latest observation: " + (trace[-1]["observation"] if trace else "No observations.")
        )
    
    LAST_ACTIVE_PROVIDER["provider"] = provider_used
    LAST_ACTIVE_PROVIDER["model"] = model_used

    return {
        "answer": final_answer,
        "trace": trace,
        "full_prompt": full_prompt_sent,
        "raw_responses": raw_responses,
        "total_tokens": total_tokens,
        "total_latency_ms": total_latency_ms,
        "total_retry_wait_ms": total_retry_wait_ms,
        "fallback_used": fallback_used,
        "tool_calls_valid_json": all_valid_json,
        "model": model_used,
        "provider": provider_used,
        "demo_mode": False,
    }

# ---------------------------------------------------------------------------
# Inspector state
# ---------------------------------------------------------------------------
_last_result = {}

def chat_handler(
    message: str,
    history,
    version: str,
    provider: str,
    temperature: float,
):
    """Handle chat messages, executing ReAct loop and recording state."""
    global _last_result
    
    if not message or not message.strip():
        return ""
    
    # Security: 1. Input length validation & capping
    valid_len, len_msg = validate_input_length(message)
    if not valid_len:
        return f"⚠️ {len_msg}"
        
    # Security: 2. Per-session sliding window rate limiting
    allowed_rate, rate_msg = global_rate_limiter.check_rate_limit("default")
    if not allowed_rate:
        return f"⏳ {rate_msg}"
        
    # Security: 3. PII Redaction before text reaches LLM & logs
    sanitized_message, pii_detected = redact_pii(message)
    
    chat_hist = []
    if history:
        for item in history:
            if isinstance(item, dict):
                chat_hist.append({
                    "role": item.get("role", "user"),
                    "content": item.get("content", ""),
                })
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                u, b = item
                if u:
                    chat_hist.append({"role": "user", "content": str(u)})
                if b:
                    chat_hist.append({"role": "assistant", "content": str(b)})
    
    try:
        result = run_react_loop(
            user_message=sanitized_message,
            chat_history=chat_hist,
            version=version,
            provider=provider.lower(),
            temperature=temperature,
        )
    except LLMUnavailableError as err:
        # No telemetry / live-turn record is written for a call that never reached an LLM.
        return (
            "🚫 **No LLM available - this request was not answered.**\n\n"
            + err.user_message().replace("\n", "\n\n")
            + "\n\nFix the provider (quota, key or model) and send the message again."
        )
    
    # Telemetry: Record run into SQLite telemetry_runs table
    tools_called = [
        step["action"]
        for step in result.get("trace", [])
        if step.get("action") and step["action"] != "finish"
    ]
    tokens = result.get("total_tokens", {})
    prompt_tokens = tokens.get("prompt_tokens", 0)
    comp_tokens = tokens.get("completion_tokens", 0)
    
    telemetry_rec = log_telemetry_run(
        version=version,
        provider=result.get("provider", provider),
        model=result.get("model", ""),
        temperature=temperature,
        top_p=0.9,
        prompt_tokens=prompt_tokens,
        completion_tokens=comp_tokens,
        latency_ms=result.get("total_latency_ms", 0),
        tool_calls=tools_called,
        demo_mode=result.get("demo_mode", False),
    )
    result["est_cost_usd"] = telemetry_rec.get("est_cost_usd", 0.0)
    _last_result = result

    # Record real-time telemetry for Evaluation Dashboard live tracking
    try:
        status_label = "✅ Success" if "error" not in result.get("answer", "")[:50].lower() else "⚠️ Error"
        record_live_turn(
            query=sanitized_message,
            version=version,
            provider=provider,
            latency_ms=result.get("total_latency_ms", 0),
            tools_called=tools_called,
            answer=result.get("answer", ""),
            status=status_label,
            iterations=len(result.get("trace", [])),
        )
    except Exception as e:
        print(f"Warning: live telemetry recording failed: {e}")

    return result["answer"]

# ---------------------------------------------------------------------------
# Inspector tab handler
# ---------------------------------------------------------------------------

def get_inspector_data():
    """Return structured visual metrics and trace for the Prompt Inspector tab."""
    global _last_result
    
    if not _last_result:
        empty_kpi = """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 20px;">
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 14px; padding: 16px;">
                <div style="font-size: 1.6rem; font-weight: 800; color: #94a3b8;">—</div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase;">⏱️ Total Latency</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 14px; padding: 16px;">
                <div style="font-size: 1.6rem; font-weight: 800; color: #94a3b8;">—</div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase;">🪙 Token Usage</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 14px; padding: 16px;">
                <div style="font-size: 1.6rem; font-weight: 800; color: #94a3b8;">—</div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase;">🔄 ReAct Steps</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 14px; padding: 16px;">
                <div style="font-size: 1.3rem; font-weight: 800; color: #94a3b8;">Ready</div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase;">🤖 Backend Engine</div>
            </div>
        </div>
        """
        return (
            "No interactions yet. Ask a question in the 💬 Chat tab first.",
            "No raw responses recorded.",
            "Send a message to see the ReAct Thought ➔ Action ➔ Observation execution timeline.",
            empty_kpi,
        )
    
    r = _last_result
    tokens = r.get("total_tokens", {})
    prompt_tokens = tokens.get("prompt_tokens", 0)
    comp_tokens = tokens.get("completion_tokens", 0)
    total_tokens = tokens.get("total_tokens", 0)
    latency_ms = r.get("total_latency_ms", 0)
    iterations = len(r.get("trace", []))
    model_name = r.get("model", "N/A")
    provider_name = r.get("provider", "N/A").upper()
    
    kpi_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 20px;">
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 14px; padding: 16px;">
            <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8;">{latency_ms:,} <span style="font-size: 0.85rem; font-weight: 500;">ms</span></div>
            <div style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;">⏱️ Total Latency</div>
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 14px; padding: 16px;">
            <div style="font-size: 1.6rem; font-weight: 800; color: #c084fc;">{total_tokens:,} <span style="font-size: 0.85rem; font-weight: 500;">({prompt_tokens}p + {comp_tokens}c)</span></div>
            <div style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;">🪙 Total Tokens</div>
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 14px; padding: 16px;">
            <div style="font-size: 1.6rem; font-weight: 800; color: #34d399;">{iterations} <span style="font-size: 0.85rem; font-weight: 500;">iterations</span></div>
            <div style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;">🔄 ReAct Steps</div>
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 14px; padding: 16px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #fbbf24; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{model_name}</div>
            <div style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;">🤖 Backend: {provider_name}</div>
        </div>
    </div>
    """
    
    trace_cards = []
    for step in r.get("trace", []):
        it_num = step.get("iteration", 1)
        thought = step.get("thought", "")
        action = step.get("action", "")
        act_input = json.dumps(step.get("action_input", {}), indent=2)
        obs = step.get("observation", "")
        
        badge_color = "#38bdf8" if action == "finish" else "#f59e0b"
        card_border = "#38bdf8" if action == "finish" else "#6366f1"
        
        trace_cards.append(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.2); border-left: 5px solid {card_border}; border-radius: 12px; padding: 16px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 800; color: #f8fafc; font-size: 1rem;">Step {it_num}</span>
                <span style="background: rgba(255,255,255,0.08); color: {badge_color}; padding: 3px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 700;">Action: {action}</span>
            </div>
            <div style="margin-bottom: 8px; font-size: 0.88rem; color: #e2e8f0; line-height: 1.5;"><strong>💡 Thought:</strong> {thought}</div>
            <div style="margin-bottom: 8px; font-size: 0.85rem; color: #94a3b8;"><strong>🔧 Parameters:</strong> <code style="font-family: monospace; background: rgba(0,0,0,0.35); color: #a5b4fc; padding: 3px 6px; border-radius: 4px;">{act_input}</code></div>
            <div style="font-size: 0.86rem; color: #34d399; line-height: 1.4;"><strong>📥 Observation:</strong> <span style="color: #cbd5e1;">{obs}</span></div>
        </div>
        """)
    
    trace_html = "".join(trace_cards) if trace_cards else "No tool calls executed."
    full_prompt = r.get("full_prompt", "N/A")
    raw_responses = "\n\n---\n\n".join(r.get("raw_responses", ["N/A"]))
    
    return full_prompt, raw_responses, trace_html, kpi_html

# ---------------------------------------------------------------------------
# Evaluation Dashboard
# ---------------------------------------------------------------------------

def load_results_csv() -> str:
    """Load eval/results.csv and format as a clean, sanitized Markdown table."""
    if not RESULTS_CSV.exists():
        return "No evaluation results found. Run evaluation to generate `eval/results.csv`."
    
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return "Results CSV is empty."
    
    headers = list(rows[0].keys())
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        cells = []
        for h in headers:
            val = str(row.get(h, "") or "")
            clean_val = val.replace("\r\n", " ").replace("\n", " ").replace("|", "/")
            if len(clean_val) > 120:
                clean_val = clean_val[:117] + "..."
            cells.append(clean_val)
        table += "| " + " | ".join(cells) + " |\n"
    
    return table

def generate_eval_charts():
    """Generate high-resolution evaluation bar charts from results.csv."""
    if not RESULTS_CSV.exists():
        return None
    
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return None
    
    import statistics
    from collections import defaultdict
    version_metrics = defaultdict(lambda: {
        "success": [], "tool_accuracy": [], "hallucination": [], "latency": []
    })
    
    def normalize_version(v: str) -> str:
        v = (v or "").strip()
        if v.startswith("V1"): return "V1 — Naive"
        if v.startswith("V2"): return "V2 — Role + Rules"
        if v.startswith("V3"): return "V3 — Few-Shot"
        if v.startswith("V4"): return "V4 — Chain-of-Thought (CoT)"
        if v.startswith("V5"): return "V5 — ReAct"
        if v.startswith("V6"): return "V6 — RAG + ReAct"
        return v
    
    for row in rows:
        v = normalize_version(row.get("version", "unknown"))
        try:
            version_metrics[v]["success"].append(float(row.get("task_success", 0)))
        except (ValueError, TypeError):
            pass
        try:
            version_metrics[v]["tool_accuracy"].append(float(row.get("tool_accuracy", 0)))
        except (ValueError, TypeError):
            pass
        try:
            version_metrics[v]["hallucination"].append(float(row.get("hallucination", 0)))
        except (ValueError, TypeError):
            pass
        try:
            lat_raw = float(row.get("latency_ms", 0))
            retry_wait = float(row.get("retry_wait_ms", 0) or 0)
            net_lat = max(0.0, lat_raw - retry_wait)
            version_metrics[v]["latency"].append(net_lat)
        except (ValueError, TypeError):
            pass
    
    CANONICAL_VERSIONS = [
        "V1 — Naive", "V2 — Role + Rules", "V3 — Few-Shot",
        "V4 — Chain-of-Thought (CoT)", "V5 — ReAct", "V6 — RAG + ReAct"
    ]
    versions = [v for v in CANONICAL_VERSIONS if v in version_metrics]
    avg = lambda lst: sum(lst) / len(lst) if lst else 0
    
    success_rates = [avg(version_metrics[v]["success"]) * 100 for v in versions]
    tool_acc = [avg(version_metrics[v]["tool_accuracy"]) * 100 for v in versions]
    halluc_rates = [avg(version_metrics[v]["hallucination"]) * 100 for v in versions]
    latencies = [
        statistics.median(version_metrics[v]["latency"]) if version_metrics[v]["latency"] else 0
        for v in versions
    ]
    
    # Modern dark tech theme for charts
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f172a")
    fig.suptitle("ShopEase AI — Systematic Prompt Engineering Benchmarks (V1–V6)", fontsize=16, fontweight="bold", color="#f8fafc", y=0.98)
    
    palette = ["#64748b", "#f59e0b", "#3b82f6", "#8b5cf6", "#6366f1", "#10b981"]
    bar_colors = palette[:len(versions)]
    
    # 1. Success Rate
    ax1 = axes[0, 0]
    ax1.set_facecolor("#1e293b")
    ax1.bar(versions, success_rates, color=bar_colors, edgecolor="#334155")
    ax1.set_title("Task Success Rate (%)", color="#38bdf8", fontweight="bold", pad=10)
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', alpha=0.15)
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", color="#ffffff")
    
    # 2. Tool Accuracy (V1/V2 shown as N/A, never 0%)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1e293b")
    plot_tool_acc = [0 if ("V1" in v or "V2" in v) else tool_acc[i] for i, v in enumerate(versions)]
    ax2.bar(versions, plot_tool_acc, color=bar_colors, edgecolor="#334155")
    ax2.set_title("Tool Call Accuracy (%)", color="#a78bfa", fontweight="bold", pad=10)
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', alpha=0.15)
    for i, v in enumerate(versions):
        if "V1" in v or "V2" in v:
            ax2.text(i, 3, "N/A", ha="center", fontweight="bold", color="#94a3b8")
        else:
            val = tool_acc[i]
            ax2.text(i, val + 2, f"{val:.0f}%", ha="center", fontweight="bold", color="#ffffff")
    
    # 3. Hallucination Rate (Lower is Better)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1e293b")
    halluc_colors = ["#ef4444" if h > 20 else "#10b981" for h in halluc_rates]
    ax3.bar(versions, halluc_rates, color=halluc_colors, edgecolor="#334155")
    ax3.set_title("Hallucination Rate (%) — [Lower is Better]", color="#f87171", fontweight="bold", pad=10)
    ax3.set_ylim(0, max(halluc_rates + [25]) * 1.3)
    ax3.grid(axis='y', alpha=0.15)
    for i, v in enumerate(halluc_rates):
        ax3.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold", color="#ffffff")
    
    # 4. Latency (Median latency excluding rate-limit backoff)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#1e293b")
    ax4.bar(versions, latencies, color=bar_colors, edgecolor="#334155")
    ax4.set_title("Median Latency (ms, excl. backoff)", color="#34d399", fontweight="bold", pad=10)
    ax4.grid(axis='y', alpha=0.15)
    for i, v in enumerate(latencies):
        ax4.text(i, v + 15, f"{v:.0f}ms", ha="center", fontweight="bold", color="#ffffff")
    
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=20, colors="#cbd5e1")
        ax.tick_params(axis='y', colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    chart_path = EVAL_DIR / "eval_charts.png"
    fig.savefig(chart_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(chart_path)


def compute_benchmark_kpi_html() -> str:
    """Calculate dynamically the latest V6 benchmark metrics from single source of truth."""
    return get_benchmark_kpi_html()


def run_live_benchmark_ui(
    target_version: str,
    provider: str,
    scope: str,
    progress=gr.Progress(track_tqdm=True),
):
    """Run benchmark test suite live with real-time progress updates in Gradio."""
    all_queries = load_test_queries()
    if "Quick" in scope:
        subset_ids = {"Q01", "Q04", "Q07", "Q10", "Q13"}
        queries_to_run = [q for q in all_queries if q["id"] in subset_ids]
    else:
        queries_to_run = all_queries
    
    if target_version == "All Versions (V1–V6)":
        versions_to_run = list_versions()
    else:
        versions_to_run = [target_version]
    
    total_tests = len(queries_to_run) * len(versions_to_run)
    bench_results = []
    start_time = time.time()
    
    for v_idx, v in enumerate(versions_to_run):
        for q_idx, q in enumerate(queries_to_run):
            curr_idx = (v_idx * len(queries_to_run)) + q_idx + 1
            progress(
                curr_idx / total_tests,
                desc=f"Running [{curr_idx}/{total_tests}] {v.split(' ')[0]}: {q['id']} ({q['category']})..."
            )
            try:
                res = evaluate_single(
                    query=q,
                    version=v,
                    provider=provider.lower(),
                    temperature=0.3,
                )
            except LLMUnavailableError as err:
                done = len(bench_results)
                stop_msg = (
                    f"🚫 **Benchmark stopped after {done}/{total_tests} cases - no LLM available.**\n\n"
                    f"{err.user_message()}\n\n"
                    "Nothing from this run was saved. Fix the provider and run again."
                )
                return stop_msg, compute_benchmark_kpi_html(), load_results_csv(), generate_eval_charts()
            bench_results.append(res)
    
    # Merge and persist to results.csv
    merge_and_save_results(bench_results)
    
    # Regenerate charts and tables
    chart_path = generate_eval_charts()
    table_md = load_results_csv()
    kpi_html = compute_benchmark_kpi_html()
    
    elapsed = round(time.time() - start_time, 1)
    success_count = sum(1 for r in bench_results if r.get("task_success"))
    pct = round((success_count / total_tests) * 100) if total_tests else 0
    status_msg = (
        f"✅ **Benchmark Complete in {elapsed}s!** Evaluated {total_tests} test cases across "
        f"{len(versions_to_run)} version(s). Overall Task Success: **{success_count}/{total_tests}** ({pct}%). "
        f"Charts and Detailed Query Matrix updated below."
    )
    
    return status_msg, kpi_html, table_md, chart_path


# ---------------------------------------------------------------------------
# Custom CSS for Modern, Premium Aesthetic
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

* {
    box-sizing: border-box;
}

:root {
    --body-background-fill: #030712 !important;
    --background-fill-primary: #030712 !important;
    --background-fill-secondary: #0b1120 !important;
    --block-background-fill: #0b1120 !important;
    --border-color-primary: rgba(99, 102, 241, 0.25) !important;
    --color-accent: #6366f1 !important;
    --body-text-color: #f8fafc !important;
    --block-label-text-color: #94a3b8 !important;
    --input-background-fill: #0f172a !important;
}

html, body, .gradio-container, .contain, .dark, body.dark, #root {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #030712 !important;
    background: #030712 !important;
    color: #f8fafc !important;
    min-height: 100vh !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 12px 24px !important;
}

/* Force dark theme on all blocks and panels */
.block, .gradio-box, .gradio-group, .panel, .tabs, .tabitem, fieldset, .form, .wrap {
    background-color: #0b1120 !important;
    background: #0b1120 !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
    color: #f8fafc !important;
}

/* Glass panel */
.glass-panel {
    background: rgba(11, 17, 32, 0.9) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 16px !important;
}

/* Tab button styling */
.tab-nav, .tabs > .tab-nav {
    background-color: #030712 !important;
    background: #030712 !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.3) !important;
}

.tab-nav button {
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 10px 18px !important;
    color: #94a3b8 !important;
    background: transparent !important;
}

.tab-nav button.selected {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
}

/* Chatbot container & message bubbles */
.chatbot, [data-testid="chatbot"], .chat-message, .bubble-wrap {
    background-color: #030712 !important;
    background: #030712 !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
}

.message-wrap {
    font-size: 0.93rem !important;
    line-height: 1.55 !important;
}

.message.user, [data-testid="user"] {
    background: linear-gradient(135deg, #4338ca, #3730a3) !important;
    color: #ffffff !important;
    border: none !important;
}

.message.bot, [data-testid="bot"] {
    background-color: #0f172a !important;
    background: #0f172a !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Inputs, Textareas & Dropdowns */
input, textarea, select, .gr-input, .gr-box, .input-wrap {
    background-color: #0f172a !important;
    background: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
}

input::placeholder, textarea::placeholder {
    color: #64748b !important;
}

/* Dropdown list items & menus */
.options, ul.options, .dropdown-menu, .secondary-wrap, .wrap-default {
    background-color: #0f172a !important;
    background: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
}

.option, li.item, .item-label {
    background-color: #0f172a !important;
    color: #f8fafc !important;
}

.option:hover, li.item:hover {
    background-color: #1e293b !important;
    color: #ffffff !important;
}

/* Accordions */
.accordion, details {
    background-color: #0b1120 !important;
    background: #0b1120 !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    color: #f8fafc !important;
}

.accordion-title, summary {
    color: #e2e8f0 !important;
}

/* Quick Chip Buttons */
.chip-btn {
    border-radius: 9999px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
    background: #0f172a !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    color: #e2e8f0 !important;
    transition: all 0.2s ease !important;
}

.chip-btn:hover {
    background: rgba(99, 102, 241, 0.35) !important;
    border-color: #818cf8 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Pulsing Online Indicator */
.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulse-ring 2s infinite;
}

@keyframes pulse-ring {
    0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
"""

# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------

QUICK_PROMPTS = [
    ("📦 Where is ORD1001?", "Where is my order ORD1001?"),
    ("🛒 Buy Headphones", "I'd like to order 2 Wireless Noise-Cancel Headphones. My name is Alex Kumar, pay with UPI."),
    ("❌ Cancel ORD1003", "Cancel my order ORD1003 please."),
    ("🔄 Return ORD1008", "I would like to return the Denim Jacket from order ORD1008."),
    ("🛍️ Browse Electronics", "Show me all electronics products available in your store."),
    ("📝 Modify ORD1016", "Change the quantity of the 4K Smart TV in order ORD1016 to 2."),
    ("📜 Return Policy", "What is your warranty and return window policy for electronics?"),
    ("🛡️ Trap: ORD9999", "Where is my order ORD9999? Track it for me."),
]

def get_header_provider_badge() -> str:
    """Generate dynamic header provider badge based on last answered reply or startup probe."""
    prov = LAST_ACTIVE_PROVIDER.get("provider", "").title()
    model = LAST_ACTIVE_PROVIDER.get("model", "")
    if prov and model:
        return f"⚡ {prov} ({model})"
    try:
        from llm_client import check_providers
        statuses = check_providers()
        groq_ok = statuses.get("groq", {}).get("ok", False)
        openrouter_ok = statuses.get("openrouter", {}).get("ok", False)
        gemini_ok = statuses.get("gemini", {}).get("ok", False)
        if groq_ok:
            return "⚡ Groq (qwen/qwen3.8-27b)"
        elif openrouter_ok:
            return "⚡ OpenRouter (llama-3.3-70b)"
        elif gemini_ok:
            return "⚡ Gemini Flash"
        else:
            return "⚠️ Providers Offline"
    except Exception:
        return "⚡ Groq (qwen/qwen3.8-27b)"

def build_ui() -> gr.Blocks:
    """Build the modern Gradio user interface."""
    prov_badge = get_header_provider_badge()
    halluc_badge = get_header_badge_html()
    
    with gr.Blocks(title="ShopEase AI — Customer Service Bot") as app:
        
        # -------------------------------------------------------------------
        # Hero Header Banner
        # -------------------------------------------------------------------
        gr.HTML(f"""
        <div style="background: linear-gradient(135deg, rgba(30, 27, 75, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 20px; padding: 22px 28px; margin-bottom: 20px; box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                <div>
                    <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        🛍️ ShopEase AI
                    </h1>
                    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.92rem; font-weight: 500;">
                        Autonomous Customer Support Agent with <strong>ReAct Multi-Step Reasoning</strong> & <strong>TF-IDF RAG Grounding</strong>
                    </p>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
                    <span class="live-indicator"><span class="pulse-dot"></span> System Online</span>
                    <span style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); color: #a5b4fc; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">{prov_badge}</span>
                    {halluc_badge}
                </div>
            </div>
        </div>
        """)
        
        with gr.Tabs():
            # ---------------------------------------------------------------
            # Tab 1: Chat Interface
            # ---------------------------------------------------------------
            with gr.Tab("💬 Interactive Chat", id="chat-tab"):
                with gr.Row():
                    # Left Sidebar: Engine Configuration & Quick Reference
                    with gr.Column(scale=1):
                        gr.HTML("<h3 style='margin: 0 0 10px 0; font-size: 1.05rem; color: #f8fafc; font-weight: 700;'>⚙️ Agent Configuration</h3>")
                        
                        provider_dropdown = gr.Dropdown(
                            choices=["Groq", "OpenRouter", "Gemini", "OpenAI", "Anthropic"],
                            value="Groq",
                            label="Model Provider",
                            info="Groq (Qwen 27B), OpenRouter (Llama 70B), Gemini (2.5 Flash), OpenAI, or Anthropic (Claude)",
                        )
                        
                        version_dropdown = gr.Dropdown(
                            choices=list_versions(),
                            value="V6 — RAG + ReAct",
                            label="Prompt Version",
                            info="Select iteration (V1 naive to V6 grounded)",
                        )
                        
                        # Live Version Explainer Card
                        version_card_box = gr.HTML(value=render_version_card("V6 — RAG + ReAct"))
                        
                        temp_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Sampling Temperature",
                            info="Lower = deterministic tool calling; Higher = creative",
                        )
                        
                        # When version changes -> auto-update card and recommended temp
                        def on_version_change(v):
                            card = render_version_card(v)
                            rec_temp = get_recommended_temp(v)
                            return card, rec_temp
                        
                        version_dropdown.change(
                            fn=on_version_change,
                            inputs=[version_dropdown],
                            outputs=[version_card_box, temp_slider],
                        )
                        
                        # Collapsible Quick Knowledge Cheatsheet
                        with gr.Accordion("🏬 Mock Database Reference", open=False):
                            gr.Markdown("""
                            **Mock Orders for Testing:**
                            - `ORD1001` — Delivered (Headphones)
                            - `ORD1002` — Shipped (Smart Watch)
                            - `ORD1003` — Processing (Espresso Beans)
                            - `ORD1008` — Delivered (Denim Jacket, returnable)
                            - `ORD1009` — Delivered (Food item, non-returnable)
                            - `ORD9999` — Fake Order (Adversarial Trap)
                            
                            **Catalog Highlights:**
                            - `UltraBook Pro 14` (₹1,099.99)
                            - `Noise-Canceling Headphones` (₹79.99)
                            - `AeroFit Running Shoes` (₹119.00)
                            """)
                        
                        with gr.Accordion("🔧 Active Store Tools (9 Tools)", open=False):
                            gr.Markdown("""
                            **📖 Query Tools:**
                            1. `get_order_status(order_id)`
                            2. `get_product_info(product_id_or_name)`
                            3. `search_policy(query)` — RAG
                            4. `list_products(category)` — Browse catalog
                            
                            **🔄 Action Tools:**
                            5. `place_order(customer_name, product_id, qty, payment_method)` — 🆕 Place real orders
                            6. `cancel_order(order_id)` — 🆕 Cancel & refund
                            7. `modify_order(order_id, product_id, new_qty)` — 🆕 Change order qty
                            8. `initiate_return(order_id, product_id)`
                            9. `calculate_refund(order_id, product_id)`
                            """)
                    
                    # Right Main Column: Chat Arena + Quick Starter Chips
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Customer Support Conversation",
                            height=520,
                            avatar_images=(None, "https://em-content.zobj.net/source/twitter/408/robot_1f916.png"),
                        )
                        
                        # Quick Starter Suggestion Chips
                        gr.HTML("<div style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin: 6px 0 4px 2px;'>⚡ Instant Test Queries (Click to Run):</div>")
                        
                        with gr.Row():
                            chip1 = gr.Button("📦 Where is ORD1001?", elem_classes=["chip-btn"])
                            chip2 = gr.Button("🛒 Buy Headphones", elem_classes=["chip-btn"])
                            chip3 = gr.Button("❌ Cancel ORD1003", elem_classes=["chip-btn"])
                            chip4 = gr.Button("🔄 Return ORD1008", elem_classes=["chip-btn"])
                        with gr.Row():
                            chip5 = gr.Button("🛍️ Browse Electronics", elem_classes=["chip-btn"])
                            chip6 = gr.Button("📝 Modify ORD1016", elem_classes=["chip-btn"])
                            chip7 = gr.Button("📜 Return Policy", elem_classes=["chip-btn"])
                            chip8 = gr.Button("🛡️ Trap: ORD9999", elem_classes=["chip-btn"])
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask about your order, return policies, product specs, or shipping...",
                                label="Your Message",
                                scale=5,
                                show_label=False,
                                container=False,
                            )
                            send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                
                # Chat logic
                def respond(message, history, version, provider, temperature):
                    if not message or not message.strip():
                        return history or [], ""
                    
                    bot_response = chat_handler(message, history, version, provider, temperature)
                    current_history = list(history) if history else []
                    current_history.append({"role": "user", "content": message})
                    current_history.append({"role": "assistant", "content": bot_response})
                    return current_history, ""
                
                send_btn.click(
                    fn=respond,
                    inputs=[msg_input, chatbot, version_dropdown, provider_dropdown, temp_slider],
                    outputs=[chatbot, msg_input],
                )
                msg_input.submit(
                    fn=respond,
                    inputs=[msg_input, chatbot, version_dropdown, provider_dropdown, temp_slider],
                    outputs=[chatbot, msg_input],
                )
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])
                
                # Wire up Quick Suggestion Chips
                chips = [chip1, chip2, chip3, chip4, chip5, chip6, chip7, chip8]
                for btn, (label, query_text) in zip(chips, QUICK_PROMPTS):
                    btn.click(
                        fn=lambda q=query_text: q,
                        outputs=[msg_input],
                    ).then(
                        fn=respond,
                        inputs=[msg_input, chatbot, version_dropdown, provider_dropdown, temp_slider],
                        outputs=[chatbot, msg_input],
                    )

            # ---------------------------------------------------------------
            # Tab 2: Prompt Inspector
            # ---------------------------------------------------------------
            with gr.Tab("🔍 Prompt & Reasoning Inspector", id="inspector-tab"):
                gr.HTML("""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <div>
                        <h3 style="margin: 0; font-size: 1.2rem; font-weight: 800; color: #f8fafc;">Execution Transparency & Telemetry</h3>
                        <p style="margin: 2px 0 0 0; color: #94a3b8; font-size: 0.88rem;">Inspect live prompt tokens, raw model answers, and step-by-step ReAct thought traces.</p>
                    </div>
                </div>
                """)
                
                refresh_btn = gr.Button("🔄 Refresh Telemetry & Traces", variant="primary")
                
                # Top KPI Grid
                kpi_box = gr.HTML(value="""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 14px; padding: 18px; margin-bottom: 16px; text-align: center; color: #94a3b8;">
                    Click <strong>'Refresh Telemetry & Traces'</strong> after sending a chat message to inspect execution telemetry.
                </div>
                """)
                
                gr.HTML("<h4 style='margin: 16px 0 8px 0; color: #e2e8f0; font-size: 1rem;'>🔄 ReAct Execution Timeline (Tool Calls & Observations)</h4>")
                trace_display = gr.HTML(value="<div style='color: #64748b;'>No execution trace loaded yet.</div>")
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h4 style='margin: 12px 0 6px 0; color: #e2e8f0; font-size: 0.95rem;'>📄 Full Prompt Sent to LLM (with RAG Context)</h4>")
                        prompt_display = gr.Textbox(
                            label="",
                            lines=14,
                            max_lines=28,
                            interactive=False,
                            show_label=False,
                        )
                    with gr.Column():
                        gr.HTML("<h4 style='margin: 12px 0 6px 0; color: #e2e8f0; font-size: 0.95rem;'>🤖 Raw LLM Output(s)</h4>")
                        raw_display = gr.Textbox(
                            label="",
                            lines=14,
                            max_lines=28,
                            interactive=False,
                            show_label=False,
                        )
                
                # Side-by-Side Prompt Version Diff
                with gr.Accordion("🔍 Side-by-Side Prompt Version Diff (difflib)", open=False):
                    gr.Markdown("Compare prompt structure and directives between any two versions (e.g. V5 ReAct vs V6 RAG):")
                    with gr.Row():
                        diff_v1 = gr.Dropdown(choices=list_versions(), value="V5 — ReAct", label="Base Version (A)", scale=2)
                        diff_v2 = gr.Dropdown(choices=list_versions(), value="V6 — RAG + ReAct", label="Target Version (B)", scale=2)
                        diff_btn = gr.Button("Compare Prompt Templates ⚖️", variant="primary", scale=1)
                    diff_output = gr.HTML(value="<div style='color: #64748b; padding: 12px;'>Select two prompt versions above and click Compare to render side-by-side diff.</div>")
                    
                    diff_btn.click(
                        fn=get_prompt_diff_html,
                        inputs=[diff_v1, diff_v2],
                        outputs=[diff_output],
                    )

                refresh_btn.click(
                    fn=get_inspector_data,
                    outputs=[prompt_display, raw_display, trace_display, kpi_box],
                )

            # ---------------------------------------------------------------
            # Tab 3: Evaluation Dashboard
            # ---------------------------------------------------------------
            with gr.Tab("📊 Evaluation Dashboard", id="eval-tab"):
                gr.HTML("""
                <div style="margin-bottom: 16px;">
                    <h3 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #f8fafc;">📊 Evaluation & Real-Time Performance Dashboard</h3>
                    <p style="margin: 3px 0 0 0; color: #94a3b8; font-size: 0.92rem;">
                        Real-time live conversation stream, interactive test suite executor, and systematic prompt refinement benchmarks (V1–V6).
                    </p>
                </div>
                """)
                
                # ===============================================================
                # PART 1: Real-Time Live Chat Telemetry (Stream Active)
                # ===============================================================
                live_kpi_box = gr.HTML(value=format_live_kpi_html())
                
                with gr.Row():
                    refresh_live_btn = gr.Button("🔄 Refresh Live Stream & Metrics", variant="primary", scale=2)
                    clear_live_btn = gr.Button("🗑️ Clear Live Session Logs", variant="secondary", scale=1)
                
                with gr.Accordion("📝 Live Interaction Stream & Conversation Logs", open=True):
                    live_table_display = gr.Markdown(value=format_live_table_markdown())
                
                gr.HTML("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 24px 0;' />")

                # ===============================================================
                # PART 1.5: SQLite Production Telemetry Aggregates
                # ===============================================================
                with gr.Accordion("🗄️ SQLite Production Telemetry Aggregates", open=False):
                    gr.Markdown("Real-time aggregates per prompt version computed directly from `data/shopease.db` (telemetry_runs):")
                    telemetry_table = gr.Markdown(value=get_telemetry_summary_markdown())
                    refresh_telem_btn = gr.Button("🔄 Refresh SQLite Telemetry Aggregates", variant="secondary")
                    refresh_telem_btn.click(fn=get_telemetry_summary_markdown, outputs=[telemetry_table])

                gr.HTML("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 24px 0;' />")

                # ===============================================================
                # PART 2: Interactive Benchmark Runner (Run Test Suite Live)
                # ===============================================================
                gr.HTML("""
                <div style="margin-bottom: 14px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <h4 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #f8fafc;">⚡ Run Benchmark Test Suite in Real-Time</h4>
                        <span style="background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.5); color: #a5b4fc; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 6px;">LIVE EXECUTION</span>
                    </div>
                    <p style="margin: 3px 0 0 0; color: #94a3b8; font-size: 0.88rem;">
                        Execute standardized customer service test queries against the model in real time. Results automatically update the benchmark database and comparison charts.
                    </p>
                </div>
                """)
                
                with gr.Row():
                    bench_version_selector = gr.Dropdown(
                        choices=["V6 — RAG + ReAct", "V5 — ReAct", "V4 — Chain-of-Thought (CoT)", "V3 — Few-Shot", "V2 — Role + Rules", "V1 — Naive", "All Versions (V1–V6)"],
                        value="V6 — RAG + ReAct",
                        label="Target Prompt Version",
                        scale=2,
                    )
                    bench_scope_selector = gr.Radio(
                        choices=["Quick Smoke Test (5 Queries: Q01, Q04, Q07, Q10, Q13)", "Full Test Suite (15 Queries: Q01–Q15)"],
                        value="Quick Smoke Test (5 Queries: Q01, Q04, Q07, Q10, Q13)",
                        label="Evaluation Scope",
                        scale=3,
                    )
                    bench_provider_selector = gr.Dropdown(
                        choices=["groq", "openrouter", "gemini", "openai", "anthropic"],
                        value="groq",
                        label="Backend Provider",
                        scale=1,
                    )
                
                run_bench_btn = gr.Button("⚡ Run Benchmark Suite Now", variant="primary")
                bench_status_display = gr.Markdown("Ready to execute live benchmark. Click above to run.")

                gr.HTML("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 24px 0;' />")

                # ===============================================================
                # PART 3: Systematic Benchmark Performance Cards & Charts (V1–V6)
                # ===============================================================
                gr.HTML("""
                <div style="margin-bottom: 14px;">
                    <h4 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #f8fafc;">📈 Systematic Benchmark Metrics & Comparison Charts</h4>
                    <p style="margin: 3px 0 0 0; color: #94a3b8; font-size: 0.88rem;">
                        Empirical comparison across standardized queries showing the iterative improvement from V1 (Zero-Shot) to V6 (RAG + ReAct).
                    </p>
                </div>
                """)
                
                bench_kpi_box = gr.HTML(value=compute_benchmark_kpi_html())
                
                reload_charts_btn = gr.Button("🔄 Reload Saved Benchmark Results & Charts", variant="secondary")
                
                chart_image = gr.Image(
                    label="Comparative Benchmark Charts (Task Success, Tool Accuracy, Hallucination Prevention, Latency)",
                    type="filepath",
                    value=str(EVAL_DIR / "eval_charts.png") if (EVAL_DIR / "eval_charts.png").exists() else None,
                )
                
                with gr.Accordion("📋 Detailed Evaluation Query Matrix (Q01–Q15)", open=False):
                    results_table = gr.Markdown(value=load_results_csv())

                # Live stream event hooks
                def refresh_live_stream():
                    return format_live_kpi_html(), format_live_table_markdown()

                def clear_live_stream():
                    clear_live_logs()
                    return format_live_kpi_html(), format_live_table_markdown()

                refresh_live_btn.click(
                    fn=refresh_live_stream,
                    outputs=[live_kpi_box, live_table_display],
                )
                clear_live_btn.click(
                    fn=clear_live_stream,
                    outputs=[live_kpi_box, live_table_display],
                )

                # Live benchmark runner hook
                run_bench_btn.click(
                    fn=run_live_benchmark_ui,
                    inputs=[bench_version_selector, bench_provider_selector, bench_scope_selector],
                    outputs=[bench_status_display, bench_kpi_box, results_table, chart_image],
                )

                # Static refresh hook
                def refresh_eval():
                    table_md = load_results_csv()
                    chart_path = generate_eval_charts()
                    kpi_html = compute_benchmark_kpi_html()
                    return kpi_html, table_md, chart_path

                reload_charts_btn.click(
                    fn=refresh_eval,
                    outputs=[bench_kpi_box, results_table, chart_image],
                )

            # ---------------------------------------------------------------
            # Tab 4: A/B Testing Arena
            # ---------------------------------------------------------------
            with gr.Tab("⚖️ A/B Testing", id="ab-tab"):
                gr.HTML("""
                <div style="margin-bottom: 16px;">
                    <h3 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #f8fafc;">⚖️ Concurrent Model A/B Testing Arena</h3>
                    <p style="margin: 3px 0 0 0; color: #94a3b8; font-size: 0.92rem;">
                        Dispatch one prompt concurrently to two versions/temperatures. Compare answers, reasoning traces, latency, and tokens side-by-side, then vote on the preferred output.
                    </p>
                </div>
                """)

                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h4 style='color: #818cf8; margin: 0 0 6px 0;'>🤖 Model Configuration A</h4>")
                        ab_ver_a = gr.Dropdown(choices=list_versions(), value="V5 — ReAct", label="Model A Version")
                        ab_temp_a = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.1, label="Model A Temperature")
                    with gr.Column():
                        gr.HTML("<h4 style='color: #34d399; margin: 0 0 6px 0;'>🤖 Model Configuration B</h4>")
                        ab_ver_b = gr.Dropdown(choices=list_versions(), value="V6 — RAG + ReAct", label="Model B Version")
                        ab_temp_b = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.1, label="Model B Temperature")

                ab_provider = gr.Dropdown(choices=["groq", "openrouter", "gemini", "openai", "anthropic"], value="groq", label="Inference Provider")

                with gr.Row():
                    ab_prompt_input = gr.Textbox(
                        placeholder="Enter any customer query to test side-by-side (e.g. Can I return my electronics order ORD1013?)...",
                        label="Evaluation Prompt",
                        scale=5,
                    )
                    ab_run_btn = gr.Button("⚡ Run A/B Comparison", variant="primary", scale=1)

                ab_status = gr.Markdown("Ready to compare. Enter a query and click Run.")

                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h4 style='color: #818cf8;'>Output A</h4>")
                        ab_ans_a = gr.Markdown(value="*Awaiting execution...*")
                        with gr.Accordion("🔍 Model A ReAct Trace", open=False):
                            ab_trace_a = gr.Textbox(label="", lines=6, interactive=False)
                    with gr.Column():
                        gr.HTML("<h4 style='color: #34d399;'>Output B</h4>")
                        ab_ans_b = gr.Markdown(value="*Awaiting execution...*")
                        with gr.Accordion("🔍 Model B ReAct Trace", open=False):
                            ab_trace_b = gr.Textbox(label="", lines=6, interactive=False)

                gr.HTML("<div style='margin-top: 14px; font-weight: 700; color: #cbd5e1;'>🗳️ Cast Your Human Preference Vote:</div>")
                with gr.Row():
                    vote_a_btn = gr.Button("👍 Model A is Better", variant="secondary")
                    vote_b_btn = gr.Button("👍 Model B is Better", variant="secondary")
                    vote_tie_btn = gr.Button("🤝 It's a Tie", variant="secondary")

                vote_status = gr.Markdown("")

                # Win-count summary
                def render_ab_summary_html() -> str:
                    s = get_ab_summary()
                    tot = s["Total"]
                    if tot == 0:
                        return """
                        <div style="text-align: center; padding: 16px; color: #94a3b8; font-style: italic; background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.1); border-radius: 8px; margin: 12px 0;">
                            no votes recorded yet
                        </div>
                        """
                    p_a = round((s["A"] / tot) * 100, 1) if tot else 0
                    p_b = round((s["B"] / tot) * 100, 1) if tot else 0
                    p_tie = round((s["Tie"] / tot) * 100, 1) if tot else 0
                    anecdotal_note = ""
                    if tot < 10:
                        anecdotal_note = f"""
                        <div style="margin-top: 10px; font-size: 0.8rem; color: #fbbf24; text-align: center; background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); border-radius: 8px; padding: 6px 12px;">
                            ⚠️ <strong>Sample Size Notice</strong>: n = {tot} (anecdotal sample, not statistically meaningful). Note: 3 existing records are legacy, non-blind.
                        </div>
                        """
                    return f"""
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0;">
                        <div style="background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3); border-radius: 10px; padding: 12px; text-align: center;">
                            <div style="font-size: 1.4rem; font-weight: 800; color: #818cf8;">{s['A']}/{tot} ({p_a}%)</div>
                            <div style="font-size: 0.76rem; color: #cbd5e1; text-transform: uppercase;">Output A Wins</div>
                        </div>
                        <div style="background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); border-radius: 10px; padding: 12px; text-align: center;">
                            <div style="font-size: 1.4rem; font-weight: 800; color: #34d399;">{s['B']}/{tot} ({p_b}%)</div>
                            <div style="font-size: 0.76rem; color: #cbd5e1; text-transform: uppercase;">Output B Wins</div>
                        </div>
                        <div style="background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.3); border-radius: 10px; padding: 12px; text-align: center;">
                            <div style="font-size: 1.4rem; font-weight: 800; color: #fbbf24;">{s['Tie']}/{tot} ({p_tie}%)</div>
                            <div style="font-size: 0.76rem; color: #cbd5e1; text-transform: uppercase;">Ties</div>
                        </div>
                        <div style="background: rgba(148,163,184,0.15); border: 1px solid rgba(148,163,184,0.3); border-radius: 10px; padding: 12px; text-align: center;">
                            <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc;">{tot}</div>
                            <div style="font-size: 0.76rem; color: #cbd5e1; text-transform: uppercase;">Total Votes</div>
                        </div>
                    </div>
                    {anecdotal_note}
                    """

                ab_summary_box = gr.HTML(value=render_ab_summary_html())

                with gr.Row():
                    export_csv_btn = gr.Button("📥 Export A/B Votes to CSV", variant="secondary")
                    export_file = gr.File(label="Downloaded CSV", interactive=False)

                # State tracking for A/B runs
                ab_state = gr.State({})

                def run_ab_comparison_ui(prompt, v_a, t_a, v_b, t_b, provider):
                    if not prompt or not prompt.strip():
                        return "Please enter a test query.", "", "", "", "Ready", {}
                    
                    ok, msg = validate_input_length(prompt)
                    if not ok:
                        return f"⚠️ {msg}", "", "", "", "Invalid Input", {}
                    
                    clean_prompt, _ = redact_pii(prompt)

                    # Step 4: Randomize which configuration is shown as Output A vs Output B (Blind Protocol)
                    import random
                    swapped = random.choice([True, False])
                    target_a_ver, target_a_temp = (v_b, t_b) if swapped else (v_a, t_a)
                    target_b_ver, target_b_temp = (v_a, t_a) if swapped else (v_b, t_b)
                    
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        f_a = executor.submit(
                            run_react_loop,
                            user_message=clean_prompt,
                            chat_history=[],
                            version=target_a_ver,
                            provider=provider.lower(),
                            temperature=target_a_temp,
                        )
                        f_b = executor.submit(
                            run_react_loop,
                            user_message=clean_prompt,
                            chat_history=[],
                            version=target_b_ver,
                            provider=provider.lower(),
                            temperature=target_b_temp,
                        )
                        try:
                            res_a = f_a.result()
                            res_b = f_b.result()
                        except LLMUnavailableError as err:
                            return (
                                "", "", "", "",
                                "🚫 No LLM available - comparison not run, nothing to vote on.\n"
                                + err.user_message(),
                                {},
                            )

                    a_config_name = f"{target_a_ver} (T={target_a_temp})"
                    b_config_name = f"{target_b_ver} (T={target_b_temp})"

                    trace_a_str = "\n".join(
                        f"[{s['iteration']}] {s['thought']} -> {s['action']} (Input: {json.dumps(s.get('action_input', {}))}) -> {s.get('observation', '')[:120]}"
                        for s in res_a.get("trace", [])
                    )
                    trace_b_str = "\n".join(
                        f"[{s['iteration']}] {s['thought']} -> {s['action']} (Input: {json.dumps(s.get('action_input', {}))}) -> {s.get('observation', '')[:120]}"
                        for s in res_b.get("trace", [])
                    )

                    saved_state = {
                        "prompt": clean_prompt,
                        "version_a": v_a,
                        "temp_a": t_a,
                        "version_b": v_b,
                        "temp_b": t_b,
                        "target_a_ver": target_a_ver,
                        "target_a_temp": target_a_temp,
                        "target_b_ver": target_b_ver,
                        "target_b_temp": target_b_temp,
                        "a_config": a_config_name,
                        "b_config": b_config_name,
                        "swapped": swapped,
                        "answer_a": res_a.get("answer", ""),
                        "answer_b": res_b.get("answer", ""),
                        "latency_a": res_a.get("total_latency_ms", 0),
                        "latency_b": res_b.get("total_latency_ms", 0),
                        "tokens_a": res_a.get("total_tokens", {}).get("total_tokens", 0),
                        "tokens_b": res_b.get("total_tokens", {}).get("total_tokens", 0),
                        "trace_a_str": trace_a_str,
                        "trace_b_str": trace_b_str,
                        "provider": provider,
                    }

                    # Blind presentation: NEVER leak version name, temperature, latency, or tokens before vote!
                    badge_a = "**Output A (Blinded)**\n\n"
                    badge_b = "**Output B (Blinded)**\n\n"

                    trace_hidden_placeholder = (
                        "🔒 *Reasoning trace and performance metrics are hidden during blind evaluation to prevent operator bias. "
                        "They will be revealed after you cast your vote below.*"
                    )

                    return (
                        badge_a + res_a.get("answer", ""),
                        trace_hidden_placeholder,
                        badge_b + res_b.get("answer", ""),
                        trace_hidden_placeholder,
                        "⚖️ Blind comparison generated! Evaluate Output A vs Output B below and cast your vote:",
                        saved_state,
                    )

                ab_run_btn.click(
                    fn=run_ab_comparison_ui,
                    inputs=[ab_prompt_input, ab_ver_a, ab_temp_a, ab_ver_b, ab_temp_b, ab_provider],
                    outputs=[ab_ans_a, ab_trace_a, ab_ans_b, ab_trace_b, ab_status, ab_state],
                )

                def vote_action(choice: str, state: dict):
                    if not state or not state.get("prompt"):
                        return "⚠️ Please run an A/B comparison query before voting.", render_ab_summary_html()
                    
                    swapped = state.get("swapped", False)
                    v_a = state["version_a"]
                    v_b = state["version_b"]
                    t_a = state["temp_a"]
                    t_b = state["temp_b"]
                    
                    # Map vote back to true configuration
                    if choice == "A":
                        winner_config = state["a_config"]
                        winning_version = state["target_a_ver"]
                    elif choice == "B":
                        winner_config = state["b_config"]
                        winning_version = state["target_b_ver"]
                    else:
                        winner_config = "Tie (Both equally acceptable)"
                        winning_version = "Tie"
                    
                    log_ab_vote(
                        prompt=state["prompt"],
                        version_a=v_a,
                        temp_a=t_a,
                        version_b=v_b,
                        temp_b=t_b,
                        answer_a=state["answer_a"],
                        answer_b=state["answer_b"],
                        vote=choice,
                        a_config=state["a_config"],
                        b_config=state["b_config"],
                        swapped=1 if swapped else 0,
                        protocol="blind",
                        tokens_a=state["tokens_a"],
                        tokens_b=state["tokens_b"],
                        latency_a_ms=state["latency_a"],
                        latency_b_ms=state["latency_b"],
                        trace_a=state["trace_a_str"],
                        trace_b=state["trace_b_str"],
                    )
                    
                    prov = state.get("provider", "Groq")
                    reveal_info = (
                        f"🎉 **Vote recorded: {choice}** (Winner: **{winning_version}**)\n\n"
                        f"--- \n"
                        f"### 🔓 Blind Protocol Reveal\n"
                        f"* **Output A was**: `{state['a_config']}` — ⏱️ {state['latency_a']:.0f} ms | 🪙 {state['tokens_a']} tokens\n"
                        f"* **Output B was**: `{state['b_config']}` — ⏱️ {state['latency_b']:.0f} ms | 🪙 {state['tokens_b']} tokens\n"
                        f"* **Protocol**: Blind randomized mapping ({'swapped' if swapped else 'direct'} presentation) | Provider: {prov}\n\n"
                        f"#### Reasoning Traces (Post-Vote):\n"
                        f"**Output A Trace**:\n```\n{state['trace_a_str']}\n```\n\n"
                        f"**Output B Trace**:\n```\n{state['trace_b_str']}\n```"
                    )
                    return reveal_info, render_ab_summary_html()

                vote_a_btn.click(
                    fn=lambda s: vote_action("A", s),
                    inputs=[ab_state],
                    outputs=[vote_status, ab_summary_box],
                )
                vote_b_btn.click(
                    fn=lambda s: vote_action("B", s),
                    inputs=[ab_state],
                    outputs=[vote_status, ab_summary_box],
                )
                vote_tie_btn.click(
                    fn=lambda s: vote_action("Tie", s),
                    inputs=[ab_state],
                    outputs=[vote_status, ab_summary_box],
                )

                def do_export_csv():
                    out_path = export_ab_log_csv()
                    return out_path

                export_csv_btn.click(fn=do_export_csv, outputs=[export_file])
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px 0 10px 0; color: #64748b; font-size: 0.85rem; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 24px;">
            ShopEase AI v2.0 • Autonomous ReAct Engine • Multi-Provider Inference (Groq / Gemini / OpenAI) • Iterative Prompt Engineering
        </div>
        """)
    
    return app

# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if os.getenv("SKIP_LLM_HEALTHCHECK", "").strip() != "1":
        print("\n[ShopEase] Checking LLM providers (one tiny request each)...")
        _status = check_providers()
        for _prov, _s in _status.items():
            print(f"  {'OK ' if _s['ok'] else 'FAIL'} {_prov}: {_s['message']}")
        if not any(s["ok"] for s in _status.values()):
            print("\n[ShopEase] WARNING: no LLM provider is reachable. The app will start, but every "
                  "request will show an error until a provider works. Run `python check_llm.py`.\n")
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue="indigo",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
        js="() => { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }",
    )
