"""Tests for llm_client fallback behaviour. No network: providers are faked."""
import pytest
import llm_client as lc


def ok(provider):
    def _call(messages, model=None, temperature=0.3, top_p=0.9, max_tokens=512):
        return {"content": "hi", "model": "m", "provider": provider,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "latency_ms": 5, "retry_wait_ms": 0, "fallback_used": False,
                "finish_reason": "stop"}
    return _call


def boom(msg):
    def _call(*a, **k):
        raise RuntimeError(msg)
    return _call


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(k, "real-looking-key")
    monkeypatch.delenv("ALLOW_PROVIDER_FALLBACK", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_ORDER", raising=False)


def test_primary_used_when_healthy(monkeypatch):
    monkeypatch.setattr(lc, "_call_groq", ok("groq"))
    r = lc.chat_completion_with_fallback([{"role": "user", "content": "x"}], "groq")
    assert r["provider"] == "groq" and r["fallback_used"] is False


def test_falls_back_to_next_real_provider(monkeypatch):
    monkeypatch.setattr(lc, "_call_groq", boom("Rate limit reached on tokens per day (TPD)"))
    monkeypatch.setattr(lc, "_call_openrouter", ok("openrouter"))
    r = lc.chat_completion_with_fallback([{"role": "user", "content": "x"}], "groq")
    assert r["provider"] == "openrouter" and r["fallback_used"] is True
    assert "tokens per day" in r["primary_error"].lower()


def test_all_fail_raises_with_real_reasons_never_fake_output(monkeypatch):
    monkeypatch.setattr(lc, "_call_groq", boom("tokens per day (TPD) limit"))
    monkeypatch.setattr(lc, "_call_openrouter", boom("429 rate limit exceeded"))
    monkeypatch.setattr(lc, "_call_anthropic", boom("401 invalid api key"))
    monkeypatch.setattr(lc, "_call_openai", boom("429 insufficient_quota"))
    monkeypatch.setattr(lc, "_call_gemini", boom("Connection timed out"))
    with pytest.raises(lc.LLMUnavailableError) as exc:
        lc.chat_completion_with_fallback([{"role": "user", "content": "x"}], "groq")
    kinds = {p: k for p, (k, _) in exc.value.errors.items()}
    assert kinds == {"groq": "quota", "openrouter": "rate_limit", "anthropic": "auth", "openai": "quota", "gemini": "network"}
    assert "No LLM provider could answer" in str(exc.value)


def test_missing_keys_are_skipped_not_called(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your_anthropic_api_key_here")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(lc, "_call_groq", boom("quota exceeded"))
    monkeypatch.setattr(lc, "_call_gemini", ok("gemini"))
    monkeypatch.setattr(lc, "_call_anthropic", boom("must not be called"))
    r = lc.chat_completion_with_fallback([{"role": "user", "content": "x"}], "groq")
    assert r["provider"] == "gemini"


def test_no_cross_provider_fallback_in_eval_mode(monkeypatch):
    monkeypatch.setenv("ALLOW_PROVIDER_FALLBACK", "0")
    monkeypatch.setattr(lc, "_call_groq", boom("tokens per day (TPD)"))
    monkeypatch.setattr(lc, "_call_openrouter", ok("openrouter"))
    with pytest.raises(lc.LLMUnavailableError) as exc:
        lc.chat_completion_with_fallback([{"role": "user", "content": "x"}], "groq")
    assert list(exc.value.errors) == ["groq"]


def test_custom_fallback_order(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_ORDER", "openai,gemini")
    assert lc.provider_order("groq") == ["groq", "openai", "gemini"]


def test_classify_error():
    assert lc.classify_error(RuntimeError("Error 429 rate limit")) == "rate_limit"
    assert lc.classify_error(RuntimeError("model `x` does not exist (404)")) == "model"
    assert lc.classify_error(RuntimeError("weird")) == "other"


def test_check_providers_reports_each(monkeypatch):
    monkeypatch.setattr(lc, "_call_groq", ok("groq"))
    monkeypatch.setattr(lc, "_call_openrouter", ok("openrouter"))
    monkeypatch.setattr(lc, "_call_anthropic", boom("credit balance too low"))
    monkeypatch.setattr(lc, "_call_openai", ok("openai"))
    monkeypatch.setattr(lc, "_call_gemini", ok("gemini"))
    s = lc.check_providers()
    assert s["groq"]["ok"] and s["openrouter"]["ok"] and not s["anthropic"]["ok"] and s["anthropic"]["kind"] == "quota"

