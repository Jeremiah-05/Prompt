import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_client import chat_completion

if __name__ == "__main__":
    for p in ['groq', 'gemini', 'openai', 'anthropic']:
        try:
            res = chat_completion([{'role': 'user', 'content': 'hi'}], provider=p)
            print(f"{p}: SUCCESS model={res.get('model')} lat={res.get('latency_ms')}ms")
        except Exception as e:
            print(f"{p}: FAILED error={str(e)[:140]}")

