"""
LLM Client — abstraction layer over Groq, Gemini, OpenAI, and Anthropic APIs.
Routes calls based on MODEL_PROVIDER env var or explicit provider parameter.
Includes token counting, latency tracking, and automatic fallback.
"""

import os
import time
import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv optional; real env vars still work
    pass


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")


def _get_provider(provider: str | None = None) -> str:
    """Resolve which provider to use."""
    p = (provider or DEFAULT_PROVIDER).lower().strip()
    if p not in ("groq", "gemini", "openai", "anthropic", "openrouter"):
        raise ValueError(f"Unknown provider: {p}. Use 'groq', 'gemini', 'openai', 'anthropic', or 'openrouter'.")
    return p


# ---------------------------------------------------------------------------
# Token counting utilities
# ---------------------------------------------------------------------------

def count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3).
    For accurate counts, use tiktoken if available.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _call_groq(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> dict:
    """Call Groq Chat Completions API.
    
    Returns:
        Dict with 'content', 'model', 'usage', 'latency_ms' keys.
    """
    from groq import Groq
    
    key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not key or key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY not set. Please add it to your .env file.")
    
    client = Groq(api_key=key, timeout=60.0)
    model_name = model or GROQ_MODEL
    
    start = time.time()
    max_retries = 4
    total_retry_wait_ms = 0
    call_latency_ms = 0
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            call_latency_ms += round((time.time() - t0) * 1000)
            break
        except Exception as e:
            err_msg = str(e).lower()
            if "tpd" in err_msg or "tokens per day" in err_msg or "credit" in err_msg:
                raise
            if ("429" in err_msg or "rate limit" in err_msg or "timed out" in err_msg or "timeout" in err_msg or "connection" in err_msg) and attempt < max_retries - 1:
                wait_sec = (attempt + 1) * 4
                total_retry_wait_ms += wait_sec * 1000
                print(f"\n[Groq retry] {type(e).__name__}: {str(e)[:80]}. Backing off for {wait_sec}s (retry {attempt+1}/{max_retries-1})...")
                time.sleep(wait_sec)
            else:
                raise
    
    choice = response.choices[0]
    usage = response.usage
    content = choice.message.content or getattr(choice.message, "reasoning", "") or ""
    
    return {
        "content": content,
        "model": model_name,
        "provider": "groq",
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
        "latency_ms": call_latency_ms,
        "retry_wait_ms": total_retry_wait_ms,
        "fallback_used": False,
        "finish_reason": choice.finish_reason,
    }


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _call_gemini(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> dict:
    """Call Google Gemini Generative Language API.
    
    Returns:
        Dict with 'content', 'model', 'usage', 'latency_ms' keys.
    """
    import google.generativeai as genai
    
    key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    if not key or key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY not set. Please add it to your .env file.")
    
    genai.configure(api_key=key)
    model_name = model or GEMINI_MODEL
    
    system_instruction = ""
    gemini_history = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            system_instruction += content + "\n"
        elif role == "user":
            gemini_history.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            gemini_history.append({"role": "model", "parts": [content]})
    
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    
    gemini_model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction.strip() if system_instruction.strip() else None,
        generation_config=generation_config,
    )
    
    start = time.time()
    
    if len(gemini_history) > 1:
        chat = gemini_model.start_chat(history=gemini_history[:-1])
        response = chat.send_message(gemini_history[-1]["parts"][0])
    elif gemini_history:
        response = gemini_model.generate_content(gemini_history[0]["parts"][0])
    else:
        response = gemini_model.generate_content("Hello")
    
    latency_ms = round((time.time() - start) * 1000)
    content = response.text if hasattr(response, "text") and response.text else ""
    
    input_text = " ".join(m.get("content", "") for m in messages)
    prompt_tokens = count_tokens_approx(input_text)
    completion_tokens = count_tokens_approx(content)
    
    return {
        "content": content,
        "model": model_name,
        "provider": "gemini",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "latency_ms": latency_ms,
        "retry_wait_ms": 0,
        "fallback_used": False,
        "finish_reason": "stop",
    }


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _call_openai(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> dict:
    """Call OpenAI Chat Completions API.
    
    Returns:
        Dict with 'content', 'model', 'usage', 'latency_ms' keys.
    """
    from openai import OpenAI
    
    key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    if not key or key == "your_openai_api_key_here":
        raise ValueError("OPENAI_API_KEY not set. Please add it to your .env file.")
    
    client = OpenAI(api_key=key)
    model_name = model or OPENAI_MODEL
    
    start = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency_ms = round((time.time() - start) * 1000)
    
    choice = response.choices[0]
    usage = response.usage
    content = choice.message.content or getattr(choice.message, "reasoning", "") or ""
    
    return {
        "content": content,
        "model": model_name,
        "provider": "openai",
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
        "latency_ms": latency_ms,
        "retry_wait_ms": 0,
        "fallback_used": False,
        "finish_reason": choice.finish_reason,
    }


# ---------------------------------------------------------------------------
# Anthropic (Claude) client
# ---------------------------------------------------------------------------

def _call_anthropic(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> dict:
    """Call Anthropic Messages API (Claude).

    Returns:
        Dict with 'content', 'model', 'usage', 'latency_ms' keys.
    """
    try:
        import importlib
        anthropic_lib = importlib.import_module("anthropic")
        Anthropic = getattr(anthropic_lib, "Anthropic")
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is not installed. Please run: pip install anthropic"
        )

    key = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    if not key or key == "your_anthropic_api_key_here":
        raise ValueError("ANTHROPIC_API_KEY not set. Please add it to your .env file.")

    client = Anthropic(api_key=key)
    model_name = model or ANTHROPIC_MODEL

    # Separate system message from conversation turns
    system_text = ""
    claude_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_text += content + "\n"
        else:
            claude_messages.append({"role": role, "content": content})

    start = time.time()
    max_retries = 4
    total_retry_wait_ms = 0
    call_latency_ms = 0
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            kwargs = {
                "model": model_name,
                "messages": claude_messages,
                "max_tokens": max_tokens,
                "extra_body": {
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
            if system_text.strip():
                kwargs["system"] = system_text.strip()
            response = client.messages.create(**kwargs)
            call_latency_ms += round((time.time() - t0) * 1000)
            break
        except Exception as e:
            err_msg = str(e).lower()
            if "credit" in err_msg or "billing" in err_msg:
                raise RuntimeError(
                    "Anthropic API error: Credit balance is too low. "
                    "Please visit https://console.anthropic.com/settings/plans to add credits."
                ) from e
            if ("429" in err_msg or "rate limit" in err_msg or "overloaded" in err_msg) and attempt < max_retries - 1:
                wait_sec = (attempt + 1) * 4
                total_retry_wait_ms += wait_sec * 1000
                print(f"\n[Anthropic 429] Rate limit hit. Backing off for {wait_sec}s (retry {attempt+1}/{max_retries-1})...")
                time.sleep(wait_sec)
            else:
                raise

    content = ""
    if response.content:
        content = "".join(block.text for block in response.content if hasattr(block, "text"))

    usage = response.usage
    return {
        "content": content,
        "model": model_name,
        "provider": "anthropic",
        "usage": {
            "prompt_tokens": usage.input_tokens if usage else 0,
            "completion_tokens": usage.output_tokens if usage else 0,
            "total_tokens": (usage.input_tokens + usage.output_tokens) if usage else 0,
        },
        "latency_ms": call_latency_ms,
        "retry_wait_ms": total_retry_wait_ms,
        "fallback_used": False,
        "finish_reason": response.stop_reason or "stop",
    }


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

def _call_openrouter(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> dict:
    """Call OpenRouter Chat Completions API.

    Returns:
        Dict with 'content', 'model', 'usage', 'latency_ms' keys.
    """
    from openai import OpenAI

    key = os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    if not key or key == "your_openrouter_api_key_here":
        raise ValueError("OPENROUTER_API_KEY not set. Please add it to your .env file.")

    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://shopease.ai",
            "X-Title": "ShopEase AI",
        },
    )
    model_name = model or OPENROUTER_MODEL

    start = time.time()
    max_retries = 3
    total_retry_wait_ms = 0
    call_latency_ms = 0
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            call_latency_ms += round((time.time() - t0) * 1000)
            break
        except Exception as e:
            err_msg = str(e).lower()
            if ("429" in err_msg or "rate limit" in err_msg) and attempt < max_retries - 1:
                wait_sec = (attempt + 1) * 3
                total_retry_wait_ms += wait_sec * 1000
                time.sleep(wait_sec)
            else:
                raise

    choice = response.choices[0]
    usage = response.usage
    content = choice.message.content or getattr(choice.message, "reasoning", "") or ""

    return {
        "content": content,
        "model": model_name,
        "provider": "openrouter",
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
        "latency_ms": call_latency_ms,
        "retry_wait_ms": total_retry_wait_ms,
        "fallback_used": False,
        "finish_reason": choice.finish_reason,
    }


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def chat_completion(
    messages: list[dict],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> dict:
    """Unified chat completion that routes to Groq, Gemini, OpenAI, Anthropic, or OpenRouter."""
    p = _get_provider(provider)
    
    if p == "groq":
        return _call_groq(messages, model, temperature, top_p, max_tokens)
    elif p == "gemini":
        return _call_gemini(messages, model, temperature, top_p, max_tokens)
    elif p == "openai":
        return _call_openai(messages, model, temperature, top_p, max_tokens)
    elif p == "anthropic":
        return _call_anthropic(messages, model, temperature, top_p, max_tokens)
    elif p == "openrouter":
        return _call_openrouter(messages, model, temperature, top_p, max_tokens)
    else:
        raise ValueError(f"Provider '{p}' not implemented.")


# ---------------------------------------------------------------------------
# Provider availability, error classification, and fallback
# ---------------------------------------------------------------------------
# Design rule: a failed LLM call is NEVER replaced by simulated output here.
# If no real provider can answer, LLMUnavailableError is raised with the real
# reason per provider, and the UI must show that error instead of a demo reply.

_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def classify_error(exc: Exception) -> str:
    """Return one of: quota, rate_limit, auth, model, network, other."""
    m = str(exc).lower()
    if any(s in m for s in ("tpd", "tokens per day", "quota", "credit", "billing", "insufficient")):
        return "quota"
    if "429" in m or "rate limit" in m or "overloaded" in m:
        return "rate_limit"
    if any(s in m for s in ("401", "403", "invalid api key", "unauthorized", "authentication", "permission")):
        return "auth"
    if ("404" in m or "model" in m) and any(s in m for s in ("not found", "does not exist", "decommission", "unsupported")):
        return "model"
    if any(s in m for s in ("connection", "timeout", "timed out", "network", "dns", "resolve")):
        return "network"
    return "other"


_KIND_HINT = {
    "quota": "daily/credit quota exhausted - wait for the reset or add credit",
    "rate_limit": "rate limited - retry in a minute",
    "auth": "API key rejected - re-copy the key into .env",
    "model": "model id not available - fix the model name",
    "network": "network error - check your connection",
    "other": "unexpected error - see details",
}


def _has_key(provider: str) -> bool:
    v = os.getenv(_KEY_ENV[provider], "")
    return bool(v) and "your_" not in v and "_here" not in v


class LLMUnavailableError(RuntimeError):
    """No real LLM could answer. Callers must surface this, never fake a reply."""

    def __init__(self, errors: dict, skipped: dict):
        self.errors = errors      # provider -> (kind, message)
        self.skipped = skipped    # provider -> reason
        super().__init__(self.user_message())

    def user_message(self) -> str:
        lines = ["No LLM provider could answer this request."]
        for prov, (kind, msg) in self.errors.items():
            lines.append(f"- {prov}: {_KIND_HINT.get(kind, kind)} [{msg[:140]}]")
        for prov, why in self.skipped.items():
            lines.append(f"- {prov}: skipped ({why})")
        return "\n".join(lines)


def provider_order(primary: str | None = None) -> list[str]:
    """Primary first, then LLM_FALLBACK_ORDER (default groq,openrouter,anthropic,openai,gemini).
    Set ALLOW_PROVIDER_FALLBACK=0 for evaluations so every row uses one provider."""
    p = _get_provider(primary)
    if os.getenv("ALLOW_PROVIDER_FALLBACK", "1").strip() == "0":
        return [p]
    configured = os.getenv("LLM_FALLBACK_ORDER", "groq,openrouter,anthropic,openai,gemini")
    others = []
    for name in (x.strip().lower() for x in configured.split(",")):
        if name in _KEY_ENV and name != p and name not in others:
            others.append(name)
    return [p] + others


def chat_completion_with_fallback(
    messages: list[dict],
    primary_provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> dict:
    """Try the primary provider, then the others that have a real key configured.

    Returns the first real LLM result (with fallback_used / primary_error set when
    a different provider answered). Raises LLMUnavailableError if none can answer.
    """
    primary = _get_provider(primary_provider)
    errors: dict = {}
    skipped: dict = {}

    for prov in provider_order(primary):
        if not _has_key(prov):
            skipped[prov] = f"{_KEY_ENV[prov]} not set"
            continue
        try:
            result = chat_completion(
                messages, prov, model if prov == primary else None,
                temperature, top_p, max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - classified and reported below
            errors[prov] = (classify_error(exc), str(exc).replace("\n", " "))
            print(f"[LLM] provider '{prov}' failed ({errors[prov][0]}): {str(exc)[:160]}")
            continue
        if prov != primary:
            result["fallback_used"] = True
            result["primary_error"] = "; ".join(f"{p}: {e[1][:120]}" for p, e in errors.items())
        return result

    raise LLMUnavailableError(errors, skipped)


def check_providers() -> dict:
    """Make one tiny real call per configured provider.
    Returns {provider: {"ok": bool, "kind": str, "message": str}} for the UI/startup check."""
    status = {}
    probe = [{"role": "user", "content": "Reply with the single word: ok"}]
    for prov in _KEY_ENV:
        if not _has_key(prov):
            status[prov] = {"ok": False, "kind": "auth", "message": f"{_KEY_ENV[prov]} not set"}
            continue
        try:
            chat_completion(probe, prov, None, 0.0, 1.0, 8)
            status[prov] = {"ok": True, "kind": "", "message": "reachable"}
        except Exception as exc:  # noqa: BLE001
            kind = classify_error(exc)
            status[prov] = {"ok": False, "kind": kind, "message": f"{_KIND_HINT[kind]} [{str(exc)[:120]}]"}
    return status
