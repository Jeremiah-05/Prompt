import os
import sys
from pathlib import Path

# Force Groq only, no fallback
os.environ["ALLOW_PROVIDER_FALLBACK"] = "0"
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from eval.run_eval import run_full_evaluation, save_results, HELDOUT_RESULTS_CSV

print("=" * 70)
print("Starting HELD-OUT Evaluation on Groq (V6, T=0.3, ALLOW_PROVIDER_FALLBACK=0)")
print("=" * 70)

results = run_full_evaluation(
    provider="groq",
    versions=["V6 — RAG + ReAct"],
    temperature=0.3,
    delay_between=2.5,
    heldout=True,
    output_path=HELDOUT_RESULTS_CSV,
)
save_results(results, HELDOUT_RESULTS_CSV)
print("=" * 70)
print(f"Held-out evaluation complete! Rows saved: {len(results)}")
print("=" * 70)
