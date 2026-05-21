# SHL Conversational Assessment Recommender — Approach Document

## 1. Problem & Approach

### Problem
Hiring managers need to find relevant SHL assessments from a catalog of 377+ Individual Test Solutions, but they typically describe needs in vague, conversational terms ("I need to hire a Java developer") rather than searching by exact product names. A conversational agent bridges this gap.

### Solution Architecture

We built a **stateless RAG (Retrieval-Augmented Generation) agent** exposed as a FastAPI service. The key insight is that a conversational recommender needs three capabilities: (1) understanding what the user needs, (2) finding relevant assessments, and (3) presenting them with grounded explanations.

**Pipeline per request:**
```
Conversation History → LLM Query Planner → Hybrid Retrieval (FAISS + Lexical) → LLM (Gemini Flash) → Catalog Validation → Response
```

**Why RAG over pure LLM?** A pure LLM approach would hallucinate assessment names and URLs. By retrieving from a curated catalog and validating every recommendation against it, we guarantee factual accuracy. The LLM's role is intent understanding and natural language generation — not information storage.

**Why stateless?** The evaluator sends the full conversation history on every request. Statelessness means zero session management, easy horizontal scaling, and no data persistence concerns. The agent infers conversational state (clarifying vs. recommending vs. refining) purely from the message history on each call.

## 2. Technical Implementation

### Data Pipeline & Hybrid Retrieval
Each assessment is embedded into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`. We use a **Hybrid Retrieval** approach to maximize Recall@10:
1. **Multi-Query Semantic Search (FAISS):** Instead of concatenating the user's message into one query (which averages out skills), we split the request into per-skill facets (e.g., "Java", "Docker", "SQL"). Each facet gets its own FAISS search, and the results are unioned.
2. **Lexical Name Index:** We built an exact-match index mapping 443 specific skill tokens to assessment names. If a user asks for "Docker", the lexical index provides a massive score boost to the "Docker (New)" assessment, bypassing the fuzziness of semantic search.
3. **LLM Query Planner:** We use Gemini as an invisible query planner before retrieval. It infers implicit needs (e.g., expanding "admin" to "Microsoft Excel" and "Microsoft Word", or adding "OPQ" for professional hires).

### Agent Logic (Gemini 2.0 Flash)
The main LLM handles five actions: **CLARIFY**, **RECOMMEND**, **REFINE**, **COMPARE**, and **REFUSE**.

Key design decisions:
- **Strong vs. Weak Off-Topic Detection:** We implemented a two-pass heuristic. Strong patterns (prompt injections, jokes) trigger an immediate refusal. Weak patterns (weather, sports) trigger a refusal only if no hiring context is present. This ensures robust handling of adversarial behavior probes.
- **Aggressive recommendation:** If the user provides a role or job description, the prompt instructs the LLM to recommend immediately — ensuring we don't over-clarify and waste turns.
- **JSON mode:** Gemini's `response_mime_type="application/json"` guarantees strictly structured output.
- **Catalog validation:** Every recommendation is cross-checked against the catalog by URL and name. Hallucinated assessments are stripped out before returning the response.

### Schema Enforcement
The response schema (`reply`, `recommendations[]`, `end_of_conversation`) is enforced at three levels:
1. **LLM prompt**: Explicit JSON schema with escaped `{ }` characters in the system prompt.
2. **Pydantic models**: Request/response validation.
3. **Exception handlers**: Catch-all wrappers return a valid fallback schema even on unhandled errors.

### Fallback Strategy
If the Gemini API fails (timeout, rate limit, quota), the agent seamlessly degrades to a **Retriever-Only Fallback Mode**. It skips the LLM generation step and returns the top 10 items from the hybrid retriever directly in the correct JSON schema. This ensures the API never crashes or returns a 500 error to the evaluator.
