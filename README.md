# HR Policy Assistant — AI Agent-Based Document Query System

An AI agent–powered application that lets users upload HR policy documents (PDF, TXT, CSV, Excel) and ask natural-language questions about them. The system uses Retrieval-Augmented Generation (RAG) and a lightweight agentic layer (planning, tool-use, validation) to produce grounded, hallucination-resistant answers.

**Live App:** [PASTE YOUR STREAMLIT CLOUD URL HERE]
**GitHub Repo:** https://github.com/CHATURESHDH/hr-policy-assistant

---

## 1. Project Overview

This project was built as a Generative AI capstone to demonstrate a full-stack RAG + Agentic AI workflow applied to enterprise document question-answering, using HR policy documents as the domain.

A user uploads a document, the system processes and indexes it, and the user can then ask questions in plain English. The system decides whether a question needs document lookup or can be answered conversationally, retrieves the most relevant content when needed, generates a grounded answer using an LLM, and validates that answer before returning it.

---

## 2. Architecture

```
┌─────────────────┐
│   User (Browser) │
└────────┬─────────┘
         │
┌────────▼─────────┐
│   Streamlit UI    │  <- file upload + chat interface
└────────┬─────────┘
         │
┌────────▼──────────────────────────────────────┐
│              INGESTION PIPELINE                 │
│  1. File type detection (PDF/TXT/CSV/Excel)     │
│  2. Text extraction (pdfplumber / pandas)        │
│  3. Chunking (LangChain RecursiveCharacterSplit) │
│  4. Embedding (SentenceTransformers MiniLM-L6)   │
│  5. Storage (ChromaDB - in-memory vector store)  │
└────────┬────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────┐
│         AGENT LAYER — LangGraph StateGraph          │
│                                                      │
│                    ┌──────────┐                     │
│                    │ planner  │                     │
│                    └────┬─────┘                     │
│                         │ (conditional edge)         │
│              ┌──────────┴──────────┐                │
│              ▼                     ▼                │
│        ┌──────────┐          ┌──────────┐           │
│        │ retrieve │          │  direct  │           │
│        └────┬─────┘          └────┬─────┘           │
│             ▼                     │                  │
│        ┌──────────┐               │                  │
│        │ generate │               │                  │
│        └────┬─────┘               │                  │
│             ▼                     │                  │
│        ┌──────────┐               │                  │
│        │ validate │               │                  │
│        └────┬─────┘               │                  │
│              ▼                    ▼                  │
│                        END                           │
└──────────────────────────────────────────────────────┘
                                            │
                                    ┌───────▼──────┐
                                    │ Final Answer  │
                                    │  to User      │
                                    └───────────────┘
```

**LLM Provider:** Groq API (model: `openai/gpt-oss-20b`)
**Embedding Model:** `all-MiniLM-L6-v2` (SentenceTransformers, sourced from Hugging Face Hub, runs locally, free)
**Vector Database:** ChromaDB (in-memory / ephemeral client)
**Agent Orchestration:** LangGraph `StateGraph` — the planner, retriever, generator, validator, and direct-reply handler are implemented as graph nodes connected by explicit (including conditional) edges, with a shared `AgentState` object carrying data between them.

---

## 3. Agent Roles

The agent is implemented as a **LangGraph `StateGraph`** with five nodes and a conditional edge, rather than a single fixed retrieve-then-generate pipeline. A shared `AgentState` (question, decision, context, answer, validity flag) flows through the graph as it executes.

| Node | Responsibility |
|---|---|
| **planner** | Reads the incoming user message and classifies it as `DIRECT` (greeting/small talk — no document lookup needed) or `RETRIEVE` (a genuine policy question requiring document search). A conditional edge (`route_after_planner`) routes execution to the appropriate branch based on this decision. |
| **retrieve** | (RETRIEVE branch) Embeds the question and performs similarity search against the vector store to fetch the top-k most relevant chunks, storing them in `state["context"]`. |
| **generate** | (RETRIEVE branch) Passes the retrieved context and question to the LLM, which generates an answer grounded only in that context. |
| **validate** | (RETRIEVE branch) A separate LLM call checks whether the generated answer is actually supported by the retrieved context. If not, the state's answer is overwritten with a safe fallback message before reaching `END`. |
| **direct** | (DIRECT branch) Handles greetings/small talk conversationally, without touching the vector store. |

This graph-based planner → retrieve → generate → validate flow (with a separate direct-reply branch) satisfies the "agent-based reasoning" requirement: the system plans its approach, conditionally uses a retrieval tool, and reflects on its own output before returning it — all made explicit through LangGraph's node/edge structure rather than implicit function calls.

---

## 4. Features

- Multi-format document upload: PDF, TXT, CSV, XLSX/XLS
- Automatic chunking with overlap to preserve context across boundaries
- Semantic (meaning-based) search, not just keyword matching
- Tabular data (CSV/Excel) converted to natural-language sentences before embedding, for better semantic search quality
- Conversational routing — greetings/small talk don't trigger unnecessary document search
- Grounded answer generation — answers only use retrieved document content
- Answer validation step to reduce hallucination
- Graceful fallback when a question isn't covered by the uploaded document
- Simple chat-based UI with persistent conversation history within a session

---

## 5. Tech Stack

| Component | Technology |
|---|---|
| UI / App framework | Streamlit |
| LLM | Groq API (`openai/gpt-oss-20b`) |
| Chunking | LangChain (`langchain-text-splitters`) |
| Agent orchestration | LangGraph (`StateGraph`, conditional edges) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`, sourced from Hugging Face Hub) |
| Vector store | ChromaDB |
| Document parsing | pdfplumber (PDF), pandas (CSV/Excel), native (TXT) |
| Language | Python 3.12 |
| Deployment | Streamlit Community Cloud |
| Version control | Git + GitHub |

---

## 6. Setup Instructions (Local)

### Prerequisites
- Python 3.12
- A free Groq API key from https://console.groq.com

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/CHATURESHDH/hr-policy-assistant.git
   cd hr-policy-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. Run the app:
   ```bash
   streamlit run app.py
   ```

6. Open the local URL shown in the terminal (typically `http://localhost:8501`)

---

## 7. Deployment

The app is deployed on **Streamlit Community Cloud**, connected directly to this GitHub repository.

**Deployment steps followed:**
1. Code pushed to a public GitHub repository (`.env` excluded via `.gitignore`)
2. Repository connected to Streamlit Cloud via GitHub sign-in
3. `GROQ_API_KEY` added as a Streamlit Cloud "Secret" (since `.env` files are not deployed)
4. App deployed with `app.py` as the entry point and `requirements.txt` for dependencies

Any push to the `main` branch on GitHub automatically triggers a redeploy on Streamlit Cloud.

---

## 8. Sample Test Documents

Three synthetic sample documents (for a fictional company, "Acme Corporation") are included in the `DATA/` folder to demonstrate the system:
- `Acme_Corp_Leave_Policy.pdf` — leave, notice period, probation policies
- `Acme_Corp_Holiday_Calendar.xlsx` — public/optional holiday schedule
- `Acme_Corp_Onboarding_FAQ.txt` — new employee onboarding Q&A

These are used purely as test data. The system is designed to work with any HR policy document a user uploads — it does not contain any hardcoded knowledge of a specific company.

---

## 9. Limitations

- **Single active document at a time**: uploading a new file replaces the previous document's index rather than merging multiple documents into one knowledge base.
- **In-memory vector store**: the vector database is ephemeral (per-session); data is not persisted between app restarts.
- **Retrieval sensitivity to phrasing**: on a small document, very generically phrased questions occasionally retrieve suboptimal chunks; this was mitigated by increasing the number of retrieved chunks (top-k), but not fully eliminated.
- **No OCR support**: scanned/image-based PDFs are not supported since text extraction relies on embedded text, not image recognition.
- **Not a substitute for official HR/legal advice**: the assistant is a document-lookup tool, not an authoritative source; answers should be verified with HR for high-stakes decisions.
- **Free-tier LLM/embedding models**: response quality is dependent on the chosen lightweight model; a larger/paid model could improve nuanced reasoning.

---

## 10. Challenges Faced (Development Log)

- **PowerShell execution policy blocking venv activation**: fixed via `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.
- **PyTorch DLL load failure on Windows**: `sentence-transformers` failed to import due to a missing Microsoft Visual C++ Redistributable; resolved by installing the redistributable and restarting the system.
- **Groq model deprecation**: initially used `llama-3.1-8b-instant`, which had been deprecated by Groq; switched to the currently supported `openai/gpt-oss-20b` model.
- **Poor retrieval quality on tabular data**: raw `DataFrame.to_string()` output (column-aligned whitespace) produced poor embeddings for CSV/Excel data; fixed by converting each row into a natural-language sentence (`"Column: value, Column: value..."`) before chunking and embedding.
- **Validator logic bug**: an f-string was accidentally written as a plain string, so the validator's prompt template never actually inserted the retrieved context — causing it to reject valid, well-grounded answers. Identified through systematic debugging (isolating retrieval vs. generation vs. validation) and fixed by correcting the string prefix.
- **ChromaDB incompatibility with Streamlit Cloud**: `chromadb.Client()` raised a `KeyError` in the Streamlit Cloud environment due to a client-caching quirk; resolved by switching to `chromadb.EphemeralClient()`.
- **Document-switching bug**: uploading a second document without refreshing the page didn't reprocess it, because the app only checked whether *a* file was uploaded rather than whether a *new* file was uploaded; fixed by tracking the last processed filename in session state and comparing it against each new upload.
- **Refactor to LangGraph**: the agent logic was initially implemented as plain Python functions with `if/else` branching. It was later refactored into a formal LangGraph `StateGraph` (with explicit nodes, a shared `AgentState`, and a conditional edge for planner routing) to make the agent architecture more standardized, explicit, and easier to extend with additional nodes in the future.

---

## 11. Future Improvements

- Support for multiple simultaneous documents with source-aware citations
- Persistent vector storage (e.g., ChromaDB with disk persistence, or a hosted vector DB)
- Section-aware chunking (splitting by document headings rather than raw character count)
- A dedicated calculator/analysis tool for numeric questions over CSV/Excel data
- Conversation memory across turns (currently each question is handled independently)
- Authentication and per-user document isolation for multi-user deployment

---

## 12. Author

Capstone Project — GenAI & Agentic AI
Developed by: Chaturesh
