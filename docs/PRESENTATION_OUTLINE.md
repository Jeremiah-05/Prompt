# ShopEase AI — Course Presentation Outline

**Project:** Autonomous Customer Service Agent with ReAct Reasoning, RAG Grounding, and Prompt Engineering Progression (V1–V6)  
**Target Duration:** 8–10 Minutes (6–8 Slides)  
**Presenter:** Student Developer / Course Candidate  

---

## Slide 1: Problem Definition & Motivation

### Title: The Customer Support Hallucination Dilemma
* **Unconstrained LLMs Invent Facts:** Zero-shot generative models hallucinate fake order statuses, make up non-existent tracking IDs (e.g., `TRK-MADE-UP`), and invent return policies.
* **Real-World Impact:** Misinforming customers erodes trust, escalates dispute ticket volumes, and causes financial losses through unauthorized return promises.
* **The Solution:** A structured, multi-step prompt engineering progression combining **ReAct reasoning** and **RAG grounding** to constrain model actions to verified store data.
* **Core Objective:** Improve task success and reliability on order operations (enforcing mandatory ownership verification, preventing fabricated data), raising task success from **46.7%** (7/15, 95% Wilson CI [24.8%–69.9%]) in V1 to **100.0%** (15/15, 95% Wilson CI [79.6%–100.0%]) in V6. Tool call accuracy is N/A for V1 and V2 (no tools defined), and rises from **86.7%** (13/15) in V3 to **100.0%** (15/15) in V6. Hallucination rate is observed at **20.0%** (3/15, [7.0%–45.2%]) in V1 vs **0.0%** (0/15, [0.0%–20.4%]) in V6.

> **Speaker Notes:**  
> "Welcome everyone. In e-commerce customer support, generative AI is a double-edged sword: models sound remarkably articulate, but without structural constraints, they comfortably fabricate order tracking codes or invent unauthorized return windows. This project demonstrates how systematic, iterative prompt engineering—transitioning from naive zero-shot prompts to structured ReAct loops and RAG retrieval—can eliminate ungrounded answers and improve task success to 100.0% across our standardized benchmark evaluation."

---

## Slide 2: System Architecture & Execution Loop

### Title: Architecture — Dynamic ReAct & RAG Loop
* **Multi-Provider Backend:** Primary inference on Groq (`qwen/qwen3.8-27b`) for low latency (~775ms/step, median end-to-end V6 latency 1,550 ms), backed by OpenRouter (Llama 3.3 70B), Gemini 2.5 Flash, Anthropic Claude Sonnet 4, and OpenAI GPT-4o-mini fallbacks.
* **Deterministic ReAct Engine:** Multi-turn cycle: `Thought:` (internal reasoning) $\rightarrow$ `Action:` (tool selection) $\rightarrow$ `Action Input:` (JSON schema arguments) $\rightarrow$ `Observation:` (execution result) $\rightarrow$ `finish`.
* **Domain Tool Registry:** 9 real-time operational tools spanning live order placement, cancellation, quantity modification, catalog browsing, order lookup, return eligibility, refund math, and policy search.
* **RAG Pipeline:** bi-gram TF-IDF vectorizer over store policies (`data/policy.md` and `data/faq.md`), injecting top-3 matching chunks directly into the prompt context.

> **Speaker Notes:**  
> "Here is our end-to-end architecture. When a customer sends a query, it enters a ReAct loop capped at 5 iterations. The model cannot answer factual questions blindly. Instead, it must first output a structured Thought, select an Action from our 9 core tools, and supply strictly validated JSON arguments. The Python backend executes the tool against our database, feeds back an Observation, and the model synthesizes a final verified answer. Policy inquiries automatically trigger our bi-gram TF-IDF RAG pipeline to ground responses in store policies."

---

## Slide 3: Iterative Prompt Engineering Progression (V1 $\rightarrow$ V6)

### Title: Systematic Progression Across 6 Prompt Generations
* **V1 (Naive Zero-Shot):** Baseline. 46.7% task success (7/15) [24.8%–69.9%], 20.0% hallucination rate (3/15) [7.0%–45.2%], 26.7% ungrounded (4/15); failed due to missing tools and generic apologies.
* **V2 (Role + Rules):** Added persona and negative constraints. Refused adversarial injections (80.0% task success, 0.0% hallucination), but lacked tools to inspect data.
* **V3 (Few-Shot Exemplars):** Injected 3 annotated tool-calling dialogues. Tool call accuracy was N/A for V1 and V2 (no tools defined) and reached 86.7% (13/15) in V3; task success 86.7% (13/15) [62.1%–96.3%].
* **V4 (Chain-of-Thought):** Explicit step-by-step reasoning decomposed product categories (14d electronics vs 30d general items); task success 86.7% (13/15), tool accuracy 93.3% (14/15), hallucination 13.3% (2/15) [3.7%–37.9%].
* **V5 (ReAct Loop):** Autonomous multi-step tool chaining enabled complex workflows (order lookup $\rightarrow$ return check $\rightarrow$ refund calculation), lifting task success to 93.3% (14/15) [70.2%–98.8%] and observing 6.7% hallucination (1/15).
* **V6 (RAG + ReAct):** Injected dynamic policy chunks and strict negative constraints: **100.0% task success (15/15) [Wilson 95% CI: 79.6%–100.0%], 100.0% tool accuracy (15/15), 0.0% hallucination (0/15) [0.0%–20.4%]**.

> **Speaker Notes:**  
> "This slide illustrates our core prompt engineering methodology. Rather than tweaking prompts randomly, each version targeted a specific empirical failure mode discovered in the previous test run. V2 fixed persona and boundary constraints, V3 taught JSON syntax via few-shot examples, V4 eliminated electronics category errors through Chain-of-Thought decomposition, V5 introduced multi-step action chaining, and V6 solved knowledge hallucination by coupling the ReAct loop with bi-gram TF-IDF RAG grounding."

---

## Slide 4: Empirical Benchmark Results

### Title: Empirical Evaluation Metrics (15 Curated Test Queries)
* **Task Success:** Scaled from **46.7% (7/15)** in V1 $\rightarrow$ **80.0% (12/15)** in V2 $\rightarrow$ **86.7% (13/15)** in V3/V4 $\rightarrow$ **93.3% (14/15)** in V5 $\rightarrow$ **100.0% (15/15)** in V6.
* **Tool Call Accuracy:** N/A in V1/V2 (no tools defined) $\rightarrow$ **86.7% (13/15)** in V3 $\rightarrow$ **93.3% (14/15)** in V4 $\rightarrow$ **80.0% (12/15)** in V5 $\rightarrow$ **100.0% (15/15)** in V6.
* **Ungrounded Rate:** Dropped from **26.7% (4/15)** in V1/V2 down to **0.0% (0/15)** in V6 as retrieval grounding replaced parametric answers.
* **Hallucination Directional Shift:** Observed at **20.0% (3/15)** in V1 and **13.3% (2/15)** in V4 down to **0.0% (0/15)** in V6.
* **Date Claim Error Supplementary Check:** 0.0% (0/15) across all stored benchmark answers; audits live model phrasing attaching order dates to shipment statuses when delivery date is null (e.g. Prompt Inspector screenshot).
* **Why V5 Hallucination Differs from V4:** Under unified ground-truth scoring, V4 has 2 hallucinations (13.3%: Q07 and Q14) whereas V5 drops to 1 hallucination (6.7%: Q14). Multi-step tool chaining in V5 resolves operational errors by fetching actual delivery dates before evaluating returns. However, on adversarial policy traps (Q14), both V4 and V5 lack policy retrieval tools, proving that action execution alone cannot replace RAG knowledge grounding.

> **Speaker Notes:**  
> "Our evaluation dataset consists of 15 standardized queries categorized into order lookups, product QA, return eligibility, policy RAG, and adversarial traps. Each query accounts for 6.7 percentage points. As seen in our benchmark matrix, V6 achieves perfection across all 15 queries. Under rigorous ground-truth scoring, V5 drops hallucinations from 13.3% down to 6.7% because multi-step action execution grounds operational order checks in database reality. However, policy traps like Q14 still fail in V5 because action loops alone lack store policy texts—which only V6's RAG integration successfully grounds."

---

## Slide 5: Live Demonstration & Real-Time Telemetry

### Title: Interactive System Walkthrough
* **Order Tracking & Action Chaining:** User inquires about order `ORD1002`; bot invokes `get_order_status` and provides real delivery date and tracking badge.
* **Category-Aware Returns:** Inquiring to return electronics from `ORD1001` (delivered >14 days ago) triggers `initiate_return` and correctly declines per 14-day policy.
* **RAG Policy Inquiries:** Questions on shipping or warranty invoke `search_policy` and return verified terms from `policy.md`.
* **Adversarial Trap Resistance:** Fake order `ORD9999` is checked against the database and truthfully reported as non-existent rather than fabricated.
* **Observability:** Gradio UI provides real-time latency, token telemetry, step-by-step reasoning accordions, and a live evaluation dashboard.

> **Speaker Notes:**  
> "In our live application, we can test all four query archetypes. Notice the transparency: every response includes an expandable reasoning accordion displaying the exact Thought, Action, and Observation. In the Prompt Inspector tab, evaluators can inspect live token consumption and system prompts. Furthermore, the Evaluation Dashboard streams real-time interaction metrics and allows one-click benchmark execution."

---

## Slide 6: Parameter Sensitivity & Fairness Evaluation

### Title: Temperature Tuning and Bias Robustness
* **Temperature Sensitivity ($T=0.0$ vs $0.3$ vs $0.7$):**
  * $T=0.3$ (Best Observed Balance): 100.0% success (15/15), 100.0% tool accuracy (15/15), 0.0% hallucination (beats $T=0.0$ by one query: 15/15 vs 14/15), median latency 1,550 ms (median 24 tokens, median 2 steps).
  * $T=0.0$: 93.3% success (14/15), 93.3% tool accuracy (14/15), 0.0% hallucination; median 26 tokens, median 2 steps; elevated median latency to 4,020 ms (cause not determined).
  * $T=0.7$: 60.0% success (9/15), 53.3% tool accuracy (8/15), 0.0% hallucination, median latency 7,114 ms; higher randomness causes JSON syntax drift (median 22 tokens, median 2 steps).
* **Fairness & Tone Robustness (5 Paired Evaluations from `eval/bias_results.csv`):**
  * **Demographic Names (PAIR-01):** Western (`John Smith`) vs South Asian (`Priya Sharma`) on `ORD1001` (owned by Alice Johnson): **EQUIVALENT** on Groq (`qwen/qwen3.8-27b`) via deterministic ownership checks.
  * **Tone Sensitivity (PAIR-02):** Polite vs urgent/frustrated: **EQUIVALENT** (both elicited ownership verification on Groq; both prompted for name and refused unverified disclosure, ensuring consistent security enforcement despite conversational tone variations).
  * **Purchasing Demographics (PAIR-03):** Tariq Al-Mansoor vs Wei Chen: **EQUIVALENT** (both executed `get_product_info` -> `place_order` identically on Groq).
  * **Linguistic Register (PAIR-04):** Formal vs slang: **EQUIVALENT** (both prompted for identity verification on Groq, ensuring consistent authorization enforcement).
  * **Gender on Policy (PAIR-05):** Female vs Male: **EQUIVALENT** (both executed `search_policy` with identical warranty policy retrieval on Groq).



> **Speaker Notes:**  
> "We also evaluated hyperparameter sensitivity and demographic bias. In our temperature experiment, $T=0.3$ provided the best empirical balance, edging out $T=0.0$ by a single query (15/15 vs 14/15) while maintaining responsive 1,550 ms latency. At $T=0.7$, tool accuracy degraded to 53.3% and success fell to 60.0% due to syntax drift. In our bias experiment, all 5 paired dimensions demonstrated EQUIVALENT behavior on Groq, with deterministic identity verification protecting unauthorized access regardless of customer demographics or tone."

---

## Slide 7: Limitations & Threats to Validity

### Title: Engineering Limitations & Real-World Constraints
* **Evaluation Sample Size:** The benchmark suite contains 15 curated queries. While representative of core edge cases, each query represents 6.7 percentage points; 95% Wilson CI for 15/15 is [79.6%–100.0%].
* **Held-Out Generalization Suite:** An unseen held-out suite of 21 queries (`eval/test_queries_heldout.json`, including injection attacks H14, H15, H19–H21) was evaluated on V6 without subsequent prompt tuning, achieving 57.1% task success (12/21) [36.5%–75.5%], 38.1% tool accuracy (8/21) [20.8%–59.1%], 0.0% hallucination (0/21) [0.0%–15.5%], and 9.5% ungrounded (2/21) [2.7%–28.9%]. A separate held-out v2 set (20 queries with verified caller names) achieved 90.0% task success and 90.0% tool accuracy with 0.0% hallucination.
* **Synthetic Seed Datasets & SQLite Storage:** Initialized from seed datasets (`orders.json` containing 21 orders, `products.json` containing 15 products) loaded into an ACID SQLite database (`data/shopease.db`). High-volume enterprise stores with thousands of SKUs require distributed relational indexing.
* **Lexical vs Dense Retrieval:** Our lexical bi-gram TF-IDF vectorizer uses exact n-gram token overlap. Synonyms or out-of-vocabulary user phrasing may occasionally miss policy chunks that dense neural embeddings (e.g. BGE/OpenAI embeddings) would capture.
* **Ground-Truth Regex Scoring:** Hallucinations and task success are evaluated against ground-truth facts from SQLite and `data/policy.md` using regex patterns for required and forbidden content under unified `eval/scoring.py`.
* **Weak Authentication:** Ownership checks in `get_order_status` use caller name matching, which serves as a prototype defense but requires token-based authentication in production.

> **Speaker Notes:**  
> "To maintain scientific rigor, we acknowledge several key limitations. First, our design benchmark comprises 15 queries, meaning a single test case shifts the success rate by 6.7%, reflected in our Wilson confidence intervals. Second, our catalog is synthetic with 21 orders. Third, our RAG pipeline utilizes bi-gram TF-IDF for simplicity and local execution without GPU dependencies; in production, dense embeddings and vector databases would provide superior semantic generalization."

---

## Slide 8: Conclusion & Future Roadmap

### Title: Summary & Future Directions
* **Key Conclusion:** Prompt engineering is an empirical discipline. Role definition, few-shot syntax guidance, Chain-of-Thought decomposition, and RAG grounding are complementary pillars required to build safe autonomous agents.
* **Future Work — Dense Hybrid Search:** Transition from pure TF-IDF to a hybrid BM25 + dense vector embedding retriever (e.g. ChromaDB / FAISS).
* **Future Work — Human-in-the-Loop Escalation:** Implement automated handoff triggers to human support agents when confidence scores fall below threshold or customer sentiment is severely negative.
* **Future Work — Multi-Turn Session Memory:** Integrate session-level state machines and vector-indexed conversation history for long multi-day customer inquiries.

> **Speaker Notes:**  
> "In conclusion, ShopEase AI proves that no hallucination was observed in 15 design queries [95% Wilson CI: 0.0%–20.4%] (and 0/21 on held-out v1, 0/20 on held-out v2; pooled 0/56 [0.0%–6.4%]) when using disciplined prompt engineering and tool grounding. By pairing ReAct multi-step reasoning with RAG, we achieved a reliable, safe assistant. Future iterations will focus on dense vector retrieval, production database connectors, and human-in-the-loop escalation. Thank you, and I look forward to your questions."

