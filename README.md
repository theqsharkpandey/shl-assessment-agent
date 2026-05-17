# SHL Assessment Conversational Agent (V2 Enterprise Architecture)

## Overview
This is a production-ready Conversational Agent built for SHL. Instead of a standard keyword search, this API enables recruiters and hiring managers to discover and compare SHL Individual Test Solutions using multi-turn natural language dialogue.

### Key Architecture Features
1. **100% Stateless FastAPI Backend**: Scales infinitely. The conversation context is passed via each request, and internal agent iterations are constrained.
2. **Abstracted LLM Layer**: Designed to run seamlessly on OpenAI (`gpt-4o`) or Google Gemini (`gemini-1.5-pro`) without codebase refactoring.
3. **Robust Semantic RAG Pipeline**: Powered by `sentence-transformers` and `FAISS` for blazingly fast cosine-similarity search.
4. **Strict Schema Constraints**: Uses Pydantic for input/output validation, guaranteeing downstream systems (or automated grading harnesses) never crash on `null` responses.
5. **Prompt-Level Guardrails**: Features dynamic 8-turn conversation capping and explicit off-topic refusal policies via system prompts.

---

## 🏗 System Architecture Diagram
```text
[ Client (Replay Harness) ] --> POST /chat (JSON) --> [ FastAPI Router (app.main) ]
                                                            |
[ FAISS Index (.faiss) ] <--- (Retrieves context) --- [ Agent Orchestrator (app.services.agent) ]
                                                            |
[ Catalog DB (JSON) ] <--- (Fetch Final Metadata) --------- | (Loop: Tool calls & Reason)
                                                            |
                                                      [ LLM Abstraction Layer ]
                                                            |
                                                  (OpenAI API / Gemini API)
```

---

## 🚀 Setup & Execution

### 1. Local Development (Python 3.11+)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure Environment:**
   Copy `.env.example` to `.env` and insert your API Key.
   ```bash
   cp .env.example .env
   # Set OPENAI_API_KEY or GEMINI_API_KEY
   ```
3. **Data Pipeline (Optional if pre-packaged):**
   ```bash
   python scripts/scraper.py     # Crawls SHL and extracts metadata
   python -m app.retrieval.indexer  # Builds FAISS index
   ```
4. **Start the API:**
   ```bash
   uvicorn app.main:app --reload
   ```

### 2. Docker Deployment
Deploy effortlessly to Render, Fly.io, or Hugging Face spaces:
```bash
docker-compose up --build
```

---

## 📊 Evaluation Strategy

To ensure high-quality outputs, run the evaluation script:
```bash
python scripts/evaluate.py
```
This utility calculates **Recall@10** by firing semantic queries directly against the FAISS index and checking if the correct assessments appear in the top 10 results. It also executes **Behavioral Probes** (e.g., ensuring the agent refuses to recommend tests when the user query is too vague on Turn 1).

---

## 💡 Engineering Tradeoffs & Rationale

* **Why FAISS + sentence-transformers instead of a dedicated Vector DB like Chroma/Pinecone?**
  For a static product catalog containing ~1000 items, spinning up a separate vector database daemon is overkill and increases deployment complexity/cost. FAISS operates entirely in-memory within the Python process, resulting in single-digit millisecond latency.
* **Why raw LLM SDKs over LangChain?**
  LangChain adds excessive abstraction. By writing our own state-machine loop in `ConversationalAgent`, we maintain absolute control over the JSON parsing, error handling, dynamic turn-capping, and history coalescence, which are critical for passing strict automated harness evaluations.
* **Why is the API Stateless?**
  Storing conversation memory on the server requires session management and Redis caches. The stateless `ChatRequest` model allows horizontal scaling without sticky sessions.

## 🔮 Future Improvements
- **Streaming Responses**: Adding `StreamingResponse` for lower perceived latency on large comparisons.
- **Reranking**: Integrating a Cross-Encoder (e.g., `ms-marco-MiniLM-L-6-v2`) after FAISS retrieval to boost precision.
