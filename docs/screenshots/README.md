# Screenshots Guide for ShopEase AI Demo & Report

This folder stores high-resolution visual evidence of the ShopEase AI web interface for the course project report and demonstration slides.

## Required Screenshots

| Filename | Location in App | What to Capture & Show |
|:---|:---|:---|
| `chat.png` | **💬 Customer Support Agent Tab** | Full view of the customer support interface, showing the header indicators (`🟢 System Online`), active prompt version dropdown (`V6 — RAG + ReAct`), quick action chips, and a completed multi-turn chat exchange (e.g. order tracking or catalog query). |
| `react_trace.png` | **💬 Customer Support Agent Tab** | Expand the **"🧠 Thought Process & Tool Execution Trace"** accordion under an assistant reply. Capture the step-by-step color-coded cards showing `Thought` $\rightarrow$ `Action` $\rightarrow$ `Observation` $\rightarrow$ `finish`. |
| `inspector.png` | **🔍 Prompt & Reasoning Inspector Tab** | Click **"🔄 Refresh Telemetry & Traces"** after a chat message. Capture the 4 KPI cards (Total Latency, Token Usage, ReAct Steps, Model) and side-by-side prompt viewer showing injected RAG context. |
| `eval_dashboard.png` | **📊 Evaluation Dashboard Tab** | Capture the full dashboard view showing: (1) Real-Time Live Chat Telemetry cards and live log table (`LIVE-001`), (2) Benchmark Suite Runner controls, and (3) Systematic Benchmark KPI cards with the 4-panel comparison bar charts. |

## Recommended Specifications
- Format: PNG (.png)
- Resolution: Minimum 1920x1080 (or crisp 2x retina captures)
- Color Mode: Dark theme (matching the Gradio UI default)

## ⚠️ Status & Outdated Captures Notice (Pending Author Recapture)

In accordance with Step 7 audit requirements, the existing captures in this directory are preserved and documented as **outdated**:
- **`chat.png`**: The current capture shows an exchange conducted under **V1** baseline prompt behavior rather than the production V6 RAG+ReAct interface.
- **`eval_dashboard.png`**: The live interaction log table displays mostly **V1** session entries (`LIVE-001`), and cards reflect early session states.
- **`react_trace.png` & `inspector.png`**: Contain earlier test execution traces.

Because browser screenshot capture without an interactive visual display cannot guarantee uncropped, verified renders and live inference requires available Groq daily token quota (currently exhausted at 200,000 TPD), these captures are preserved with exact recapture instructions for the author.

## Author Action Items / Steps for Recapture After Quota Reset
When daily Groq quota resets (or using paid credits):
1. Run `python scripts/reset_demo_db.py` to reseed products (15) and orders (21), and clear live demo logs.
2. Launch `python app.py` and navigate to `http://localhost:7860/` in Chrome.
3. Conduct 4 live interactions on Groq (`qwen/qwen3.8-27b`):
   - Order tracking: *"Where is my order ORD1002?"* as Bob Stevens
   - Policy inquiry: *"What is the warranty and return policy on electronics?"*
   - Returns: *"I want to return the Denim Jacket from ORD1008"* as Henry Brown
   - Adversarial trap: *"Where is order ORD9999?"*
4. Capture and replace:
   - `chat.png`: Customer Support tab with completed chat and provider footer.
   - `react_trace.png`: Expanded Thought Process accordion showing ReAct steps.
   - `inspector.png`: Prompt Inspector tab showing telemetry and injected context.
   - `eval_dashboard.png`: Evaluation Dashboard tab showing live telemetry cards, live log table, and benchmark charts.
5. Confirm visual inspection:
   - Prompt dropdown displays `V6 — RAG + ReAct`.
   - Provider footer confirms `Groq (qwen/qwen3.8-27b)`.
   - Zero `DEMO` badges visible.
   - Evaluation dashboard KPI cards and charts match canonical CSV benchmarks (93% Task Success, 93% Tool Accuracy, 6.7% Hallucination).
