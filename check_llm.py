"""
check_llm.py - diagnostic script to verify reachability and quota for LLM providers.
Run from your project root:   python check_llm.py
Standard library only. Makes one tiny request per provider.
"""
import json
import os
import urllib.request
import urllib.error
from pathlib import Path


def load_env(path=".env"):
    p = Path(path)
    if not p.exists():
        print(f"!! {path} NOT FOUND in {Path.cwd()} - run this from the project root")
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def usable(key):
    v = os.getenv(key, "")
    return bool(v) and "your_" not in v and "_here" not in v


def post(url, headers, body):
    h = {"User-Agent": "ShopEase-Check/1.0", **headers}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()[:200]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def explain(status):
    return {
        200: "OK - this provider works",
        401: "BAD KEY - key invalid or has stray characters",
        403: "FORBIDDEN - key lacks access",
        404: "NOT FOUND - wrong model id or endpoint",
        429: "QUOTA / RATE LIMIT - daily or per-minute limit reached",
        None: "NETWORK ERROR - no connection or blocked",
    }.get(status, "see message below")


load_env()
print(f"DEMO_MODE = {os.getenv('DEMO_MODE')!r}   (must be empty or 0 for real runs)")
print(f"MODEL_PROVIDER = {os.getenv('MODEL_PROVIDER')!r}\n")

chat = {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
checks = [
    ("Groq", "GROQ_API_KEY",
     "https://api.groq.com/openai/v1/chat/completions",
     lambda k: {"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
     {**chat, "model": "qwen/qwen3.8-27b"}),
    ("OpenAI", "OPENAI_API_KEY",
     "https://api.openai.com/v1/chat/completions",
     lambda k: {"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
     {**chat, "model": "gpt-4o-mini"}),
]

for name, env, url, hdr, body in checks:
    if not usable(env):
        print(f"[{name}] {env} missing or still a placeholder")
        continue
    status, msg = post(url, hdr(os.environ[env]), body)
    print(f"[{name}] HTTP {status} -> {explain(status)}")
    if status != 200:
        print(f"         {msg}")

if usable("GEMINI_API_KEY"):
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.5-flash:generateContent?key={os.environ['GEMINI_API_KEY']}")
    status, msg = post(url, {"Content-Type": "application/json"},
                       {"contents": [{"parts": [{"text": "ping"}]}]})
    print(f"[Gemini] HTTP {status} -> {explain(status)}")
    if status != 200:
        print(f"         {msg[:200].replace(os.environ['GEMINI_API_KEY'], '***')}")
else:
    print("[Gemini] GEMINI_API_KEY missing or still a placeholder")
