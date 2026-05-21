# SHL Assessment Recommender Agent

A **conversational AI agent** that helps hiring managers and recruiters discover the right SHL assessments through natural dialogue. Built as a stateless FastAPI service with RAG (Retrieval-Augmented Generation).

## Architecture

```
User → POST /chat → Agent Controller → FAISS Retriever → Gemini LLM → Validated Response
                                              ↑
                                     377 SHL Assessments
                                     (pre-indexed vectors)
```

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API** | FastAPI | Stateless REST endpoints |
| **LLM** | Gemini 2.0 Flash | Intent analysis, response generation |
| **Embeddings** | sentence-transformers (MiniLM-L6-v2) | Semantic encoding |
| **Vector Store** | FAISS (IndexFlatIP) | Cosine similarity search |
| **Catalog** | 377 Individual Test Solutions | Ground truth data |

### How It Works

1. **User sends conversation history** → stateless `POST /chat`
2. **Agent decomposes the conversation into facets** — each named skill becomes its own query, plus an LLM query-planner adds inferred needs (e.g. "admin" → MS Word, "healthcare" → medical terminology, most roles → personality + cognitive)
3. **Hybrid retrieval** unions multi-query FAISS semantic search with a lexical name index (exact skill matches) and a soft test-type boost, producing a ~40-item candidate pool
4. **Gemini analyzes intent** and selects the best 1-10 assessments from the pool
5. **All recommendations are validated** against the catalog (no hallucination)
6. **Schema-compliant JSON** is always returned

## Quick Start

### Prerequisites
- Python 3.11+
- [Gemini API Key](https://aistudio.google.com/apikey) (free)

### Setup

```bash
# Clone and install
cd shl-assessment-agent
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### `GET /health`
```json
{"status": "ok"}
```

#### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for hiring Java developers"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are the best SHL assessments for evaluating Java developers...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

### Interactive Testing

Visit `http://localhost:8000/docs` for the Swagger UI.

## Evaluation

```bash
# Start the server first, then:
python eval/evaluate.py --base-url http://localhost:8000
```

Runs 10 conversation traces testing:
- Schema compliance on every response
- All recommended URLs exist in the catalog
- Off-topic refusal behavior
- Vague query clarification
- Multi-turn refinement support

## Deployment (Render)

1. Push to GitHub
2. Connect to Render
3. Set `GEMINI_API_KEY` environment variable
4. Deploy — the `render.yaml` handles everything

## Project Structure

```
shl-assessment-agent/
├── app/
│   ├── main.py          # FastAPI endpoints
│   ├── agent.py          # Core agent logic (RAG pipeline)
│   ├── retriever.py      # FAISS semantic search
│   ├── catalog.py        # Catalog store (in-memory)
│   ├── models.py         # Pydantic request/response schemas
│   ├── prompts.py        # System prompts and templates
│   └── config.py         # Configuration and constants
├── data/
│   ├── catalog.json      # 377 scraped SHL assessments
│   └── faiss_index/      # Pre-built FAISS index
├── eval/
│   └── evaluate.py       # Evaluation script
├── docs/
│   └── approach.md       # 2-page approach document
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```
