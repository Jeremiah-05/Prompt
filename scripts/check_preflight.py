"""Step 1 Preflight check script."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import check_providers

def main():
    s = check_providers()
    groq = s.get("groq", {})
    print(f"GROQ STATUS: ok={groq.get('ok')} message={groq.get('message')} kind={groq.get('kind')}")
    for prov, status in s.items():
        print(f"  {prov}: ok={status.get('ok')} message={status.get('message')}")

if __name__ == "__main__":
    main()
