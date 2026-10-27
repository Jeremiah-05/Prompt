"""
ShopEase AI — Telemetry & Observability Engine
Stores every execution in SQLite (data/shopease.db):
- telemetry_runs: version, provider, model, temperature, top_p, prompt/completion tokens,
  latency_ms, est_cost_usd, tool_calls, demo_mode.
- ab_log: A/B testing side-by-side votes, tokens, and latencies.
"""

import sqlite3
import json
import time
import csv
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "data" / "shopease.db"

# Model pricing in USD per 1,000,000 tokens
MODEL_PRICING = {
    # Groq Qwen
    "qwen/qwen3.8-27b": {"prompt": 0.05, "completion": 0.08},
    "qwen-2.5-32b": {"prompt": 0.05, "completion": 0.08},
    # Gemini
    "gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30},
    "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
    # OpenAI
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    # Anthropic Claude
    "claude-sonnet-4-20250514": {"prompt": 3.00, "completion": 15.00},
    "claude-sonnet-5-5": {"prompt": 3.00, "completion": 15.00},
    "claude-haiku-5-5": {"prompt": 0.80, "completion": 4.00},
    "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-haiku-20241022": {"prompt": 0.80, "completion": 4.00},
    # OpenRouter
    "meta-llama/llama-3.3-70b-instruct": {"prompt": 0.12, "completion": 0.30},
    # Default fallback pricing
    "default": {"prompt": 0.10, "completion": 0.20},
}



def get_db_connection() -> sqlite3.Connection:
    """Create a connection to SQLite database."""
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_telemetry_tables():
    """Initialize telemetry_runs and ab_log tables in SQLite."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS telemetry_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        version TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        temperature REAL NOT NULL,
        top_p REAL NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        total_tokens INTEGER NOT NULL,
        latency_ms REAL NOT NULL,
        est_cost_usd REAL NOT NULL,
        tool_calls TEXT NOT NULL,
        demo_mode INTEGER DEFAULT 0
    )
    """)
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS ab_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        prompt TEXT NOT NULL,
        version_a TEXT NOT NULL,
        temp_a REAL NOT NULL,
        version_b TEXT NOT NULL,
        temp_b REAL NOT NULL,
        answer_a TEXT NOT NULL,
        answer_b TEXT NOT NULL,
        vote TEXT NOT NULL,
        tokens_a INTEGER DEFAULT 0,
        tokens_b INTEGER DEFAULT 0,
        latency_a_ms REAL DEFAULT 0,
        latency_b_ms REAL DEFAULT 0,
        trace_a TEXT DEFAULT '',
        trace_b TEXT DEFAULT '',
        query TEXT DEFAULT ''
    )
    """)
    
    # Safe migration: add any missing columns if ab_log was created under older schema
    existing_cols = [col[1] for col in c.execute("PRAGMA table_info(ab_log)").fetchall()]
    for col_name, col_type in [
        ("tokens_a", "INTEGER DEFAULT 0"),
        ("tokens_b", "INTEGER DEFAULT 0"),
        ("latency_a_ms", "REAL DEFAULT 0"),
        ("latency_b_ms", "REAL DEFAULT 0"),
        ("trace_a", "TEXT DEFAULT ''"),
        ("trace_b", "TEXT DEFAULT ''"),
        ("query", "TEXT DEFAULT ''"),
        ("a_config", "TEXT DEFAULT ''"),
        ("b_config", "TEXT DEFAULT ''"),
        ("swapped", "INTEGER DEFAULT 0"),
        ("protocol", "TEXT DEFAULT 'legacy, non-blind'"),
    ]:
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE ab_log ADD COLUMN {col_name} {col_type}")
    
    conn.commit()
    conn.close()


# Ensure tables exist on import
init_telemetry_tables()


def calculate_cost_usd(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated API cost in USD based on model pricing table."""
    model_key = (model or "").lower()
    rates = MODEL_PRICING.get("default")
    for k, v in MODEL_PRICING.items():
        if k in model_key:
            rates = v
            break
            
    prompt_cost = (prompt_tokens / 1_000_000.0) * rates["prompt"]
    completion_cost = (completion_tokens / 1_000_000.0) * rates["completion"]
    return round(prompt_cost + completion_cost, 7)


def log_telemetry_run(
    version: str,
    provider: str,
    model: str,
    temperature: float,
    top_p: float,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    tool_calls: list[str] | str,
    demo_mode: bool = False,
) -> dict:
    """Log an agent run into SQLite telemetry table.
    
    Returns the logged record dict including estimated cost.
    """
    total_tokens = prompt_tokens + completion_tokens
    est_cost = 0.0 if demo_mode else calculate_cost_usd(provider, model, prompt_tokens, completion_tokens)
    
    if isinstance(tool_calls, (list, tuple)):
        tool_calls_str = json.dumps(tool_calls)
    else:
        tool_calls_str = str(tool_calls)
        
    ts = datetime.now(timezone.utc).isoformat()
    demo_int = 1 if demo_mode else 0
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
    INSERT INTO telemetry_runs (
        timestamp, version, provider, model, temperature, top_p,
        prompt_tokens, completion_tokens, total_tokens, latency_ms,
        est_cost_usd, tool_calls, demo_mode
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts, version, provider, model, temperature, top_p,
        prompt_tokens, completion_tokens, total_tokens, latency_ms,
        est_cost, tool_calls_str, demo_int
    ))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    
    return {
        "id": row_id,
        "timestamp": ts,
        "version": version,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "est_cost_usd": est_cost,
        "tool_calls": tool_calls,
        "demo_mode": demo_mode,
    }


def get_version_aggregates() -> list[dict]:
    """Calculate per-version aggregates from telemetry_runs excluding demo_mode runs."""
    conn = get_db_connection()
    c = conn.cursor()
    rows = c.execute("""
    SELECT 
        version,
        COUNT(*) as total_runs,
        AVG(latency_ms) as avg_latency_ms,
        AVG(prompt_tokens) as avg_prompt_tokens,
        AVG(completion_tokens) as avg_completion_tokens,
        AVG(total_tokens) as avg_total_tokens,
        SUM(est_cost_usd) as total_cost_usd
    FROM telemetry_runs
    WHERE demo_mode = 0
    GROUP BY version
    ORDER BY version ASC
    """).fetchall()
    conn.close()
    
    aggregates = []
    for r in rows:
        aggregates.append({
            "version": r["version"],
            "total_runs": r["total_runs"],
            "avg_latency_ms": round(r["avg_latency_ms"] or 0, 1),
            "avg_prompt_tokens": round(r["avg_prompt_tokens"] or 0),
            "avg_completion_tokens": round(r["avg_completion_tokens"] or 0),
            "avg_total_tokens": round(r["avg_total_tokens"] or 0),
            "total_cost_usd": round(r["total_cost_usd"] or 0, 5),
        })
    return aggregates


def get_telemetry_summary_markdown() -> str:
    """Format per-version telemetry aggregates as a Markdown table for the Dashboard."""
    aggs = get_version_aggregates()
    if not aggs:
        return "*No telemetry runs recorded yet.*"
    
    lines = [
        "| Version | Total Runs | Avg Latency | Avg Prompt Tokens | Avg Completion Tokens | Total Cost (INR) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for a in aggs:
        cost_inr = a['total_cost_usd'] * 83.0
        lines.append(
            f"| **{a['version']}** | {a['total_runs']} | {a['avg_latency_ms']:.1f} ms | "
            f"{a['avg_prompt_tokens']} | {a['avg_completion_tokens']} | ₹{cost_inr:.4f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A/B Testing Telemetry & Logging
# ---------------------------------------------------------------------------

def log_ab_vote(
    prompt: str,
    version_a: str,
    temp_a: float,
    version_b: str,
    temp_b: float,
    answer_a: str,
    answer_b: str,
    vote: str,
    a_config: str = "",
    b_config: str = "",
    swapped: int = 0,
    protocol: str = "blind",
    tokens_a: int = 0,
    tokens_b: int = 0,
    latency_a_ms: float = 0.0,
    latency_b_ms: float = 0.0,
    trace_a: str = "",
    trace_b: str = "",
) -> int:
    """Log an A/B user vote ('A', 'B', or 'Tie') into SQLite ab_log under blind protocol."""
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
    INSERT INTO ab_log (
        timestamp, prompt, query, version_a, temp_a, version_b, temp_b,
        answer_a, answer_b, vote, a_config, b_config, swapped, protocol,
        tokens_a, tokens_b, latency_a_ms, latency_b_ms, trace_a, trace_b
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts, prompt, prompt, version_a, temp_a, version_b, temp_b,
        answer_a, answer_b, vote, a_config, b_config, int(swapped), protocol,
        int(tokens_a), int(tokens_b), float(latency_a_ms), float(latency_b_ms), trace_a, trace_b
    ))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id


def log_ab_comparison(
    prompt: str,
    version_a: str,
    temp_a: float,
    version_b: str,
    temp_b: float,
    answer_a: str,
    answer_b: str,
    vote: str = "",
    tokens_a: int = 0,
    tokens_b: int = 0,
    latency_a_ms: float = 0.0,
    latency_b_ms: float = 0.0,
    trace_a: str = "",
    trace_b: str = "",
    a_config: str = "",
    b_config: str = "",
    swapped: int = 0,
    protocol: str = "blind",
) -> int:
    """Log an A/B comparison run into SQLite ab_log with empty/uncast vote."""
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
    INSERT INTO ab_log (
        timestamp, prompt, query, version_a, temp_a, version_b, temp_b,
        answer_a, answer_b, vote, tokens_a, tokens_b, latency_a_ms, latency_b_ms,
        trace_a, trace_b, a_config, b_config, swapped, protocol
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts, prompt, prompt, version_a, temp_a, version_b, temp_b,
        answer_a, answer_b, vote, int(tokens_a), int(tokens_b),
        float(latency_a_ms), float(latency_b_ms), trace_a, trace_b,
        a_config, b_config, int(swapped), protocol
    ))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id


def get_ab_summary() -> dict:
    """Get win counts from ab_log, counting only recorded non-empty votes."""
    conn = get_db_connection()
    c = conn.cursor()
    rows = c.execute("SELECT vote, COUNT(*) as cnt FROM ab_log WHERE vote IS NOT NULL AND TRIM(vote) != '' GROUP BY vote").fetchall()
    legacy_rows = c.execute("SELECT COUNT(*) as cnt FROM ab_log WHERE vote IS NOT NULL AND TRIM(vote) != '' AND (protocol LIKE '%legacy%' OR protocol = '' OR protocol IS NULL)").fetchone()
    legacy_cnt = legacy_rows["cnt"] if legacy_rows else 0
    conn.close()
    
    counts = {"A": 0, "B": 0, "Tie": 0, "Total": 0, "Legacy": legacy_cnt}
    for r in rows:
        v = r["vote"]
        if v in counts:
            counts[v] = r["cnt"]
        counts["Total"] += r["cnt"]
    return counts


def get_ab_win_report() -> str:
    """Report win counts reading only recorded votes with count/n, never percentage-only."""
    s = get_ab_summary()
    if s["Total"] == 0:
        return "no votes recorded yet"
    tot = s["Total"]
    p_a = round((s["A"] / tot) * 100, 1)
    p_b = round((s["B"] / tot) * 100, 1)
    p_tie = round((s["Tie"] / tot) * 100, 1)
    note = " (anecdotal sample, not statistically meaningful)" if tot < 10 else ""
    return (
        f"Model A: {s['A']}/{tot} ({p_a}%), Model B: {s['B']}/{tot} ({p_b}%), Ties: {s['Tie']}/{tot} ({p_tie}%), "
        f"Total Votes: {tot}{note}. Legacy non-blind votes: {s.get('Legacy', 0)}"
    )


def export_ab_log_csv(filepath: Path | str | None = None) -> str:
    """Export ab_log table to a CSV file and return the absolute path."""
    if filepath is None:
        filepath = Path(__file__).parent / "eval" / "ab_test_log.csv"
    else:
        filepath = Path(filepath)
        
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    conn = get_db_connection()
    c = conn.cursor()
    rows = c.execute("SELECT * FROM ab_log ORDER BY id DESC").fetchall()
    conn.close()
    
    headers = ["id", "timestamp", "prompt", "version_a", "temp_a", "version_b", "temp_b", "answer_a", "answer_b", "vote", "a_config", "b_config", "swapped", "protocol"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow([r[h] if h in r.keys() else "" for h in headers])
            
    return str(filepath)
