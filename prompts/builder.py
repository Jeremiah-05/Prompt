"""
Prompt Builder — loads versioned prompt templates and fills placeholders.
Supports V1 (naive) through V6 (RAG + ReAct).
"""

import os
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

# Map version keys to filenames (supports both new front-matter v1.md..v6.md and legacy filenames)
PROMPT_VERSIONS = {
    "V1 — Naive": "v1.md",
    "V2 — Role + Rules": "v2.md",
    "V3 — Few-Shot": "v3.md",
    "V4 — Chain-of-Thought (CoT)": "v4.md",
    "V5 — ReAct": "v5.md",
    "V6 — RAG + ReAct": "v6.md",
}

# Legacy fallback mapping and aliases (ensures both legacy V4 labels resolve to existing v4 template)
LEGACY_PROMPT_VERSIONS = {
    "V1 — Naive": "v1_naive.md",
    "V2 — Role + Rules": "v2_role_rules.md",
    "V3 — Few-Shot": "v3_fewshot.md",
    "V4 — Chain-of-Thought (CoT)": "v4_cot.md",
    "V4 — CoT": "v4_cot.md",
    "V4 — Chain-of-Thought": "v4_cot.md",
    "V5 — ReAct": "v5_react.md",
    "V6 — RAG + ReAct": "v6_rag_react.md",
}

# Aliases mapping legacy or alternative labels to the canonical version
VERSION_ALIASES = {
    "V4 — CoT": "V4 — Chain-of-Thought (CoT)",
    "V4 — Chain-of-Thought": "V4 — Chain-of-Thought (CoT)",
    "V4": "V4 — Chain-of-Thought (CoT)",
}

def resolve_version_key(version: str) -> str:
    """Normalize version string or alias to canonical version key."""
    if not version:
        return version
    v_clean = version.strip()
    return VERSION_ALIASES.get(v_clean, v_clean)

# Recommended temperature per version
RECOMMENDED_TEMPS = {
    "V1 — Naive": 0.7,
    "V2 — Role + Rules": 0.5,
    "V3 — Few-Shot": 0.3,
    "V4 — Chain-of-Thought (CoT)": 0.3,
    "V4 — CoT": 0.3,
    "V4 — Chain-of-Thought": 0.3,
    "V5 — ReAct": 0.3,
    "V6 — RAG + ReAct": 0.3,
}

# Tool definitions in JSON schema format for system prompt injection
TOOL_DEFINITIONS = [
    {
        "name": "get_order_status",
        "description": "Look up order details by order ID. Requires customer_name for ownership verification.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID (e.g., ORD1001)"},
                "customer_name": {"type": "string", "description": "Full name of customer for ownership verification"}
            },
            "required": ["order_id", "customer_name"]
        }
    },
    {
        "name": "get_product_info",
        "description": "Get product details by product ID or product name.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id_or_name": {"type": "string", "description": "Product ID (e.g., PROD001) or product name"}
            },
            "required": ["product_id_or_name"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search store policy documents using RAG retrieval. Returns relevant policy chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Policy search query (e.g., 'return window', 'shipping')"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "initiate_return",
        "description": "Check return eligibility and initiate a return for a specific product in an order. Requires customer_name.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID"},
                "product_id": {"type": "string", "description": "Product ID to return"},
                "customer_name": {"type": "string", "description": "Full name of customer for ownership verification"}
            },
            "required": ["order_id", "product_id", "customer_name"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund amount for a product in an order. Requires customer_name.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID"},
                "product_id": {"type": "string", "description": "Product ID"},
                "customer_name": {"type": "string", "description": "Full name of customer for ownership verification"}
            },
            "required": ["order_id", "product_id", "customer_name"]
        }
    },
    {
        "name": "place_order",
        "description": "Place a new order for a product. Creates a real order with status 'processing'. Validates stock and deducts inventory.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string", "description": "Customer's full name"},
                "product_id": {"type": "string", "description": "Product ID to order (e.g., PROD001)"},
                "qty": {"type": "integer", "description": "Quantity (default 1, max 10)"},
                "payment_method": {"type": "string", "description": "Payment: credit_card, debit_card, paypal, upi, net_banking"}
            },
            "required": ["customer_name", "product_id"]
        }
    },
    {
        "name": "cancel_order",
        "description": "Cancel an existing order. Only processing/shipped orders can be cancelled. Requires customer_name.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID to cancel (e.g., ORD1003)"},
                "customer_name": {"type": "string", "description": "Full name of customer for ownership verification"}
            },
            "required": ["order_id", "customer_name"]
        }
    },
    {
        "name": "modify_order",
        "description": "Modify the quantity of a product in an existing order. Only processing orders can be modified. Requires customer_name.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID to modify"},
                "product_id": {"type": "string", "description": "Product ID to change quantity for"},
                "new_qty": {"type": "integer", "description": "New quantity (1-10)"},
                "customer_name": {"type": "string", "description": "Full name of customer for ownership verification"}
            },
            "required": ["order_id", "product_id", "new_qty", "customer_name"]
        }
    },
    {
        "name": "list_products",
        "description": "Browse the full product catalog, optionally filtered by category (electronics, apparel, grocery).",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Optional filter: 'electronics', 'apparel', or 'grocery'. Empty for all."}
            },
            "required": []
        }
    },
    {
        "name": "finish",
        "description": "End the ReAct loop and provide the final answer to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Final answer to the user"}
            },
            "required": ["answer"]
        }
    }
]


def parse_prompt_file(filepath: Path) -> tuple[dict, str]:
    """Parse a prompt file with optional YAML front-matter into (frontmatter_dict, body)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    frontmatter = {}
    body = content

    # Check for YAML front-matter delineated by --- at start
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            raw_fm = parts[1].strip()
            body = parts[2].strip()
            # Simple, reliable line-by-line key: value YAML parser (avoids external pyyaml dependency)
            for line in raw_fm.split("\n"):
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    # Try numeric conversion
                    if v.replace(".", "", 1).isdigit():
                        v = float(v) if "." in v else int(v)
                    frontmatter[k] = v
            return frontmatter, body

    # Fallback for legacy files without front-matter: strip comment headers (# )
    lines = content.split("\n")
    body_lines = []
    in_header = True
    for line in lines:
        if in_header and line.startswith("# "):
            continue
        else:
            in_header = False
            body_lines.append(line)

    return frontmatter, "\n".join(body_lines).strip()


def load_prompt_frontmatter(version: str) -> dict:
    """Load metadata front-matter for a prompt version."""
    v = resolve_version_key(version)
    filename = (
        PROMPT_VERSIONS.get(v)
        or PROMPT_VERSIONS.get(version)
        or LEGACY_PROMPT_VERSIONS.get(v)
        or LEGACY_PROMPT_VERSIONS.get(version)
    )
    if not filename:
        raise ValueError(f"Unknown prompt version: {version}. Available: {list(PROMPT_VERSIONS.keys())}")
    filepath = PROMPTS_DIR / filename
    fm, _ = parse_prompt_file(filepath)
    return fm


def load_prompt_template(version: str) -> str:
    """Load prompt template body from .md file, stripping front-matter or comments."""
    v = resolve_version_key(version)
    filename = PROMPT_VERSIONS.get(v) or PROMPT_VERSIONS.get(version)
    filepath = PROMPTS_DIR / filename if filename else None
    if not filepath or not filepath.exists():
        filename = LEGACY_PROMPT_VERSIONS.get(v) or LEGACY_PROMPT_VERSIONS.get(version)
        filepath = PROMPTS_DIR / filename if filename else None

    if not filepath or not filepath.exists():
        raise ValueError(f"Unknown prompt version: {version}. Available: {list(PROMPT_VERSIONS.keys())}")

    _, body = parse_prompt_file(filepath)
    return body


def generate_strategy_table() -> str:
    """Generate Markdown strategy table (Version | Technique | Strength) from front-matter."""
    rows = [
        "| Version | Technique | Strength |",
        "|:---|:---|:---|",
    ]
    for v_name in list_versions():
        fm = load_prompt_frontmatter(v_name)
        ver_label = fm.get("version", v_name)
        technique = fm.get("technique", "N/A")
        strength = fm.get("strength", "N/A")
        rows.append(f"| **{ver_label}** | {technique} | {strength} |")
    return "\n".join(rows)


def get_prompt_diff_html(v1_name: str, v2_name: str) -> str:
    """Generate side-by-side HTML diff between any two prompt versions using difflib."""
    import difflib
    
    t1 = load_prompt_template(v1_name).splitlines(keepends=True)
    t2 = load_prompt_template(v2_name).splitlines(keepends=True)
    
    differ = difflib.HtmlDiff(wrapcolumn=60)
    diff_table = differ.make_table(
        t1,
        t2,
        fromdesc=f"{v1_name} Template",
        todesc=f"{v2_name} Template",
        context=True,
        numlines=3,
    )
    
    # Wrap in dark-theme styling container
    styled_diff = f"""
    <div style="max-height: 480px; overflow: auto; border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">
        <style>
            table.diff {{ width: 100%; border-collapse: collapse; background: #030712; color: #cbd5e1; }}
            .diff_header {{ background: #0f172a; color: #64748b; padding: 4px 8px; font-weight: 700; text-align: right; border-right: 1px solid rgba(255,255,255,0.1); }}
            .diff_next {{ display: none; }}
            td.diff_add {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
            td.diff_chg {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
            td.diff_sub {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
            .diff th {{ background: #0f172a; color: #a5b4fc; padding: 8px; text-align: left; font-weight: 700; border-bottom: 1px solid rgba(99, 102, 241, 0.3); }}
        </style>
        {diff_table}
    </div>
    """
    return styled_diff


def format_chat_history(history: list[dict]) -> str:
    """Format chat history list into a string for prompt injection.
    
    Args:
        history: List of dicts with 'role' and 'content' keys.
    
    Returns:
        Formatted string of conversation history.
    """
    if not history:
        return "(No previous conversation)"
    
    formatted = []
    for msg in history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)


def build_prompt(
    version: str,
    user_message: str,
    chat_history: list[dict] | None = None,
    rag_context: str = "",
    react_trace: str = "",
) -> str:
    """Build the full prompt string for a given version.
    
    Args:
        version: Prompt version key (e.g., "V6 — RAG + ReAct")
        user_message: Current user message
        chat_history: Previous messages as list of {role, content} dicts
        rag_context: Retrieved RAG context (policy/FAQ chunks), if any
        react_trace: Current ReAct trace (Thought/Action/Observation so far in this turn)
    
    Returns:
        Complete formatted prompt string ready for LLM.
    """
    template = load_prompt_template(version)
    
    history_str = format_chat_history(chat_history or [])
    
    # Fill placeholders
    prompt = template.replace("{user_message}", user_message)
    prompt = prompt.replace("{chat_history}", history_str)
    prompt = prompt.replace("{rag_context}", rag_context if rag_context else "(No policy context retrieved)")
    
    # Append current ReAct trace if continuing a loop
    if react_trace:
        prompt += f"\n\n--- Current ReAct Trace ---\n{react_trace}"
    
    return prompt


def get_compact_tool_definitions() -> list[dict]:
    """Return compacted JSON schema for tool definitions, reducing prompt overhead while preserving exact behavior."""
    compact = []
    for t in TOOL_DEFINITIONS:
        props = {}
        for p_name, p_spec in t["parameters"]["properties"].items():
            p_dict = {"type": p_spec["type"]}
            if p_name in ("payment_method", "category", "qty", "new_qty"):
                p_dict["description"] = p_spec.get("description", "")
            props[p_name] = p_dict
        compact.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": {
                "type": "object",
                "properties": props,
                "required": t["parameters"]["required"]
            }
        })
    return compact


def get_system_message(version: str) -> str:
    """Get the system-level message based on prompt version.
    
    For V1-V2, returns a minimal system message.
    For V3+, includes tool definitions in the system message.
    """
    import json
    
    currency_rule = "CURRENCY RULE: All store prices, order totals, and refunds are in Indian Rupees (INR, ₹). Always use the ₹ symbol (e.g. ₹1,099.99, ₹79 deduction). Never use dollar ($) signs.\n\n"
    
    if version in ("V1 — Naive",):
        return f"You are a helpful customer service assistant.\n\n{currency_rule}"
    
    if version in ("V2 — Role + Rules",):
        return (
            "You are ShopEase AI, a customer-support assistant for ShopEase General Store. "
            "Be concise and helpful. Never invent information.\n\n"
            f"{currency_rule}"
        )
    
    # V3+ include compacted tool schemas in system message
    tool_desc = json.dumps(get_compact_tool_definitions(), separators=(",", ":"))
    
    base = (
        "You are ShopEase AI, a customer-support assistant for ShopEase General Store.\n\n"
        f"{currency_rule}"
        "You operate in a ReAct loop: Thought → Action → Action Input → Observation → ... → finish.\n\n"
        f"Available tools (JSON schema):\n{tool_desc}\n\n"
        "Output format per step:\n"
        "Thought: <your reasoning>\n"
        "Action: <tool_name>\n"
        "Action Input: <JSON arguments>\n\n"
        "Wait for Observation before your next Thought. When done, use Action: finish."
    )
    
    return base


def list_versions() -> list[str]:
    """Return list of available prompt version names."""
    return list(PROMPT_VERSIONS.keys())


def get_recommended_temp(version: str) -> float:
    """Get recommended temperature for a prompt version."""
    v = resolve_version_key(version)
    return RECOMMENDED_TEMPS.get(v, RECOMMENDED_TEMPS.get(version, 0.3))
