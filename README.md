# Telecom LLM Assistant

AI Customer Support Assistant for a telecommunications company.  

## Implementation Details

| Technology | File | Description |
|---|---|---|
| RAG | `src/rag/retriever.py` | Chunking, embeddings, and semantic search. |
| Multi-turn Dialogue | `src/dialog/manager.py` | History management, context window, and sessions. |
| Prompt Engineering | `src/prompts/templates.py` | Active prompts for system behavior and intent classification, plus prepared templates for future flows. |
| ReAct Agent | `src/agents/react_agent.py` | LangGraph-based tool-calling agent for KB search, pricing, compatibility, and support guidance. |
| Metrics | `src/metrics/evaluator.py` | LLM-as-a-judge, prompt A/B testing. |
| Tests | `tests/test_scenarios.py` | Live integration tests for intents, hallucinations, dialogue flow, and safety. |

## Quick Start

```bash
# 1. Clone and install dependencies
git clone https://github.com/lexaisnotdead/telecom-assistant
cd telecom-assistant
pip install -r requirements.txt

# 2. Set up API key
cp .env.example .env
# Open .env and insert OPENAI_API_KEY=sk-...

# 3. Run demo (non-interactive)
python3 main.py --mode demo

# 4. Interactive RAG chat
python3 main.py --mode chat

# 5. ReAct agent with tools
python3 main.py --mode agent

# 6. Evaluate response quality
python3 main.py --mode eval

# 7. Run live integration tests
pytest tests/ -v
```

These tests call OpenAI APIs and require `OPENAI_API_KEY`, network access, and an initialized local Chroma database.

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
                      GPT-5.4-mini
                          │
                          ▼
                  Customer Response
```

## Running Modes

**`--mode demo`** — Pre-scripted dialogue for demonstration (HoReCa scenario).  
**`--mode chat`** — Interactive chat with dialogue history.  
**`--mode agent`** — LangGraph ReAct-style agent that autonomously decides which tools to use.  
**`--mode eval`** — Automated quality assessment with metrics.  

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
- **OpenAI** — LLM and embeddings provider.
- **pytest** — Scenario-based testing.
