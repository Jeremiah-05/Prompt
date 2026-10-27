import os
import sys
import time
from pathlib import Path

# Force Groq only, strictly no fallback
os.environ["ALLOW_PROVIDER_FALLBACK"] = "0"
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from eval.run_eval import run_full_evaluation, save_results

runs = [
    ("Run 1", ROOT_DIR / "eval" / "results_run1.csv"),
    ("Run 2", ROOT_DIR / "eval" / "results_run2.csv"),
    ("Run 3", ROOT_DIR / "eval" / "results_run3.csv"),
]

for label, out_csv in runs:
    print("\n" + "=" * 70)
    print(f"Starting {label} on Groq (V6, T=0.3, ALLOW_PROVIDER_FALLBACK=0)")
    print(f"Destination: {out_csv}")
    print("=" * 70)
    
    results = run_full_evaluation(
        provider="groq",
        versions=["V6 — RAG + ReAct"],
        temperature=0.3,
        delay_between=3.0,
        heldout=False,
        output_path=out_csv,
    )
    save_results(results, out_csv)
    print(f"{label} complete! Rows saved: {len(results)}")
    time.sleep(5.0)

print("\nAll 3 repeatability runs finished!")
