---
title: Telecom Assistant
emoji: 📞
colorFrom: blue
colorTo: slate
sdk: gradio
sdk_version: "6.10.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# AI Customer Support Assistant

AI Customer Support Assistant for a telecommunications company.

The project now uses:
- Gemini for chat, routing, and evaluation
- Local embeddings for RAG
- ChromaDB for vector storage
- A local Python virtual environment for development

## Implementation Details

| Technology | File | Description |
|---|---|---|
| RAG | `src/rag/retriever.py` | Chunking, local embeddings, and semantic search. |
| Multi-turn Dialogue | `src/dialog/manager.py` | History management, context window, sessions, and Gemini chat. |
| Prompt Engineering | `src/prompts/templates.py` | Active prompts for system behavior and intent classification, plus prepared templates for future flows. |
| ReAct Agent | `src/agents/react_agent.py` | LangGraph-based tool-calling agent for KB search, pricing, compatibility, and support guidance. |
| Metrics | `src/metrics/evaluator.py` | Gemini-based LLM-as-a-judge, prompt A/B testing. |
| Tests | `tests/test_scenarios.py` | Live integration tests for intents, hallucinations, dialogue flow, and safety. |

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/lexaisnotdead/ai-telecom-assistant
cd telecom-assistant

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Add GOOGLE_API_KEY in .env

# 5. Run the demo
python main.py --mode demo

# 6. Run the Gradio web app locally
python app.py

# 7. Interactive chat
python main.py --mode chat

# 8. ReAct agent with tools
python main.py --mode agent

# 9. Evaluate response quality
python main.py --mode eval

# 10. Run tests
pytest -q
```

These tests call Gemini APIs and require `GOOGLE_API_KEY`, network access, and an initialized local Chroma database.

## Local Setup Notes

- If `python3 main.py` fails with missing modules, make sure you activated `.venv` first.
- The recommended workflow is always `source .venv/bin/activate` before running the app or tests.
- The project includes a lightweight fallback embedding implementation, but the recommended setup is still `sentence-transformers` inside the virtual environment.

## Environment Variables

The app reads configuration from `.env`:

- `GOOGLE_API_KEY` - required for Gemini calls
- `GEMINI_MODEL` - defaults to `gemini-2.5-flash`
- `EMBEDDING_MODEL` - defaults to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `PERSIST_DIR` - defaults to `./chroma_db`
- `DATA_DIR` - defaults to `./data`
- `ENABLE_SELF_CORRECTION` - defaults to `false` to reduce Gemini quota usage
- `ADMIN_PASSWORD` - optional password for the admin upload tab

## Architecture

```
Customer Query
      │
      ▼
 Intent Classifier (few-shot)
      │
      ├── plan_info ──────┐
      ├── technical_issue │
      ├── billing         ▼
      └── ...       RAG Retriever
                          │
                     top-4 chunks
                          │
                          ▼
                   Prompt Assembly
                 (context + history)
                          │
                          ▼
                    Gemini 2.5 Flash
                          │
                          ▼
                  Customer Response
```

## Running Modes

**`--mode demo`** — Pre-scripted dialogue for demonstration (HoReCa scenario).  
**`--mode chat`** — Interactive chat with dialogue history.  
**`--mode agent`** — LangGraph ReAct-style agent that autonomously decides which tools to use.  
**`--mode eval`** — Automated quality assessment with metrics.  

## Free Stack

This project is designed to run on a mostly free stack:

- Gemini free tier for generation and evaluation
- Local sentence-transformers embeddings
- ChromaDB on local disk
- Hugging Face Spaces free CPU tier for deployment

## Deployment Plan

Recommended free deployment path:

1. Create a Hugging Face account.
2. Create a new Space using the Python or Gradio template.
3. Add `GOOGLE_API_KEY` as a Space secret.
4. Set `GEMINI_MODEL`, `EMBEDDING_MODEL`, `PERSIST_DIR`, and `DATA_DIR` as needed.
5. Add a small web wrapper around `DialogManager` if you want a browser UI.
6. Let the Space build the dependencies from `requirements.txt`.
7. Verify chat, agent, and evaluation flows after the first deploy.

## Hugging Face Spaces

The repository now includes a minimal Gradio entrypoint in [`app.py`](./app.py).

To deploy it on a free Hugging Face Space:

1. Create a new Space and choose the Gradio template.
2. Set the Space entrypoint to `app.py`.
3. Add `GOOGLE_API_KEY` as a Space secret.
4. Optionally set `GEMINI_MODEL`, `EMBEDDING_MODEL`, `PERSIST_DIR`, and `DATA_DIR`.
5. Push the repository and let Spaces build from `requirements.txt`.
6. Open the Space URL and test the `Chat` and `Agent` tabs.
7. Use the `Reindex knowledge base` button if you change documents in `data/`.
8. Use the `Admin` tab to upload `.txt` files or paste new knowledge base text.

## Docker

The repository also includes a minimal [`Dockerfile`](./Dockerfile) for container-based deploys.

Local build and run:

```bash
docker build -t telecom-assistant .
docker run --rm -p 7860:7860 --env-file .env telecom-assistant
```

This is useful if you want to deploy the same image to a container platform or to a Hugging Face Space using a Docker template.

## Quality Metrics

Evaluation via LLM-as-a-judge based on 4 criteria:
- **Faithfulness** — Absence of hallucinations (facts sourced only from the knowledge base).
- **Relevance** — Directness and accuracy of the answer to the query.
- **Completeness** — Fullness of the provided information.
- **Tone** — Appropriateness of the tone for a business client.

## Future Roadmap

Next steps for project expansion:
1. Add reranking (CrossEncoder) to improve RAG retrieval.
2. Add LangSmith for full-stack tracing of all calls.
3. Implement hybrid search (Vector + BM25).
4. Add a web interface using FastAPI and basic HTML.

## Tech Stack

- **LangChain** — LLM and agent orchestration.
- **LangGraph** — Stateful tool-calling agent graph.
- **ChromaDB** — Vector database (local/serverless).
- **Google Gemini** — LLM provider.
- **Sentence Transformers** — Local embeddings provider.
- **python-dotenv** — Local environment loading for development.
- **pytest** — Scenario-based testing.
