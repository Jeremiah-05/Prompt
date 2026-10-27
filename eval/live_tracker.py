"""
Live Tracker for ShopEase AI — Real-time telemetry, live conversation logging,
and live session performance tracking for the Evaluation Dashboard.
"""

import json
import time
from datetime import datetime
from pathlib import Path

LIVE_LOG_FILE = Path(__file__).parent / "live_logs.json"

_LIVE_LOGS: list[dict] = []


def _load_persisted_logs():
    global _LIVE_LOGS
    if LIVE_LOG_FILE.exists():
        try:
            with open(LIVE_LOG_FILE, "r", encoding="utf-8") as f:
                _LIVE_LOGS = json.load(f)
        except Exception:
            _LIVE_LOGS = []


def _save_persisted_logs():
    try:
        LIVE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Keep latest 100 live interactions
        with open(LIVE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(_LIVE_LOGS[-100:], f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save live logs: {e}")


# Initialize from disk if present
_load_persisted_logs()


import re

def strip_html(text: str) -> str:
    """Strip raw HTML tags from text to ensure clean plaintext display."""
    clean = re.sub(r"<[^>]+>", "", str(text or ""))
    return " ".join(clean.replace("\n", " ").split())


def record_live_turn(
    query: str,
    version: str,
    provider: str,
    latency_ms: int,
    tools_called: list[str],
    answer: str,
    status: str = "Success",
    iterations: int = 1,
) -> dict:
    """Record a real live user interaction from the chat interface."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    clean_answer = strip_html(answer)
    answer_preview = (clean_answer[:90] + "...") if len(clean_answer) > 90 else clean_answer
    
    clean_query = strip_html(query)
    query_preview = (clean_query[:65] + "...") if len(clean_query) > 65 else clean_query

    entry = {
        "id": f"LIVE-{len(_LIVE_LOGS) + 1:03d}",
        "time": timestamp,
        "query": query_preview,
        "full_query": query,
        "version": version.split(" — ")[0] if " — " in version else version,
        "full_version": version,
        "provider": provider.upper(),
        "latency_ms": latency_ms,
        "iterations": iterations,
        "tools_called": tools_called,
        "tools_str": ", ".join(tools_called) if tools_called else "None (Direct)",
        "answer_preview": answer_preview,
        "status": status,
    }
    
    _LIVE_LOGS.append(entry)
    _save_persisted_logs()
    return entry


def get_live_logs() -> list[dict]:
    """Return all recorded live interactions in current session."""
    return list(_LIVE_LOGS)


def clear_live_logs() -> None:
    """Clear recorded live interactions."""
    global _LIVE_LOGS
    _LIVE_LOGS = []
    if LIVE_LOG_FILE.exists():
        try:
            LIVE_LOG_FILE.unlink()
        except Exception:
            pass


def get_live_metrics() -> dict:
    """Calculate aggregated telemetry metrics across all live chat interactions."""
    if not _LIVE_LOGS:
        return {
            "total_queries": 0,
            "avg_latency": 0,
            "total_tools": 0,
            "success_rate": 100.0,
        }
    
    total = len(_LIVE_LOGS)
    avg_latency = round(sum(item["latency_ms"] for item in _LIVE_LOGS) / total)
    total_tools = sum(len(item.get("tools_called", [])) for item in _LIVE_LOGS)
    successes = sum(1 for item in _LIVE_LOGS if "Success" in item.get("status", ""))
    success_rate = round((successes / total) * 100, 1)

    return {
        "total_queries": total,
        "avg_latency": avg_latency,
        "total_tools": total_tools,
        "success_rate": success_rate,
    }


def format_live_kpi_html() -> str:
    """Return dark-mode responsive HTML KPI cards for live chat metrics."""
    m = get_live_metrics()
    total = m["total_queries"]
    latency = m["avg_latency"]
    tools = m["total_tools"]
    rate = m["success_rate"]

    pulse_indicator = """
    <span style="display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 3px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: #34d399;">
        <span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
        STREAM ACTIVE
    </span>
    """

    return f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 16px; padding: 18px 20px; margin-bottom: 18px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.15rem; font-weight: 800; color: #f8fafc;">🔴 Real-Time Live Chat Telemetry</span>
                {pulse_indicator}
            </div>
            <span style="font-size: 0.8rem; color: #94a3b8;">Updated automatically on every message</span>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 14px;">
                <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8;">{total}</div>
                <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">💬 Live Queries Handled</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 14px;">
                <div style="font-size: 1.6rem; font-weight: 800; color: #34d399;">{rate}%</div>
                <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">🏆 Live Success Rate</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 12px; padding: 14px;">
                <div style="font-size: 1.6rem; font-weight: 800; color: #c084fc;">{tools}</div>
                <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">🔧 Live Tool Executions</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 14px;">
                <div style="font-size: 1.6rem; font-weight: 800; color: #fbbf24;">{latency} <span style="font-size: 0.85rem; font-weight: 500;">ms</span></div>
                <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">⚡ Live Avg Latency</div>
            </div>
        </div>
    </div>
    """


def format_live_table_markdown() -> str:
    """Format live interactions into a clean markdown table."""
    if not _LIVE_LOGS:
        return (
            "> ℹ️ **No live chat interactions recorded yet in this session.**\n\n"
            "Send a message to the bot in the **💬 Customer Support Agent** tab, and it will immediately appear here with real-time latency, tools invoked, and status!"
        )
    
    # Reverse so newest are at the top
    rows = list(reversed(_LIVE_LOGS))
    
    table = "| ID | Time | Query | Version | Provider | Tools Used | Latency | Status | Response Snippet |\n"
    table += "|---|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        table += (
            f"| `{r['id']}` "
            f"| {r['time']} "
            f"| {r['query']} "
            f"| `{r['version']}` "
            f"| {r['provider']} "
            f"| `{r['tools_str']}` "
            f"| {r['latency_ms']}ms "
            f"| {r['status']} "
            f"| {r['answer_preview']} |\n"
        )
    return table
