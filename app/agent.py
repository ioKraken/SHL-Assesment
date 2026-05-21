"""Core agent logic — the brain of the SHL Assessment Recommender.

This module orchestrates:
1. Intent analysis from conversation history
2. Retrieval of relevant assessments
3. LLM-powered response generation
4. Schema-compliant output construction
"""

import json
import re
import logging
import traceback

from app.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_RECOMMENDATIONS, TEST_TYPE_MAP
from app.models import ChatRequest, ChatResponse, Recommendation, Message
from app.catalog import catalog_store, Assessment
from app.retriever import retriever
from app.prompts import build_system_prompt, format_catalog_for_context

logger = logging.getLogger("shl-agent.agent")

# Configure Gemini. The import is guarded so that an SDK/import problem
# degrades us to the retriever-only fallback instead of crashing the service.
try:
    import google.generativeai as genai
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully")
    else:
        logger.warning("GEMINI_API_KEY not set — running in retriever-only mode")
except Exception as e:  # ImportError or configuration error
    genai = None
    logger.warning(f"google.generativeai unavailable ({e}) — retriever-only mode")


def _llm_available() -> bool:
    return (
        genai is not None
        and bool(GEMINI_API_KEY)
        and GEMINI_API_KEY != "your_gemini_api_key_here"
    )


_TYPE_KEYWORDS = {
    "K": [
        "technical", "coding", "programming", "knowledge", "skills test",
        "code test", "software", "developer test", "coding test",
        "build apps", "web development", "database", "sql", "api",
        "full stack", "front end", "back end", "devops", "cloud",
        "algorithm", "data structure", "scripting",
    ],
    "P": [
        "personality", "opq", "motivation", "cultural fit",
        "behavioral traits", "personality test", "work style",
        "values", "temperament", "character", "dependability", "reliability",
    ],
    "A": [
        "cognitive", "aptitude", "reasoning", "numerical", "verbal",
        "inductive", "deductive", "ability", "analytical",
        "problem solving", "critical thinking", "iq", "mental ability",
        "logical", "abstract reasoning",
    ],
    "B": [
        "situational", "behavioral", "judgment", "sjt", "behaviour",
        "situational judgment", "scenario", "decision making",
        "interpersonal", "conflict resolution",
    ],
    "S": [
        "simulation", "inbox", "work sample", "simulate", "role play",
        "in-tray", "case study", "practical exercise",
    ],
    "C": ["competency", "competencies", "competence"],
    "D": ["development", "360", "feedback", "coaching", "growth"],
}

# Common SHL acronyms → a name fragment the catalog search can resolve.
_NAME_ALIASES = {
    "opq": "opq",
    "gsa": "global skills",
    "adept": "adept",
    "mq": "motivation questionnaire",
    "verify": "verify",
    "dsi": "dependability and safety",
}

_COMPARE_MARKERS = ("difference", "differ", "compare", "comparison",
                    " vs ", " vs.", "versus", " or ")


def _extract_test_type_boost(messages: list[Message]) -> list[str]:
    """Soft test-type hints (used as a re-rank boost, never a hard filter)."""
    all_text = " ".join(m.content.lower() for m in messages if m.role == "user")
    return [code for code, kws in _TYPE_KEYWORDS.items()
            if any(kw in all_text for kw in kws)]


def _detect_named_assessments(text: str) -> list[int]:
    """Best-effort detection of catalog items explicitly named by the user.

    Only used to PIN items into the candidate pool for comparison turns,
    so it scans capitalized phrases / acronyms rather than ordinary words.
    """
    ids, seen = [], set()
    candidates = re.findall(
        r"\b[A-Z][A-Za-z0-9+.#/-]*(?:\s+[A-Z][A-Za-z0-9+.#/-]*){0,3}\b", text
    )
    for cand in candidates:
        cand = cand.strip()
        if len(cand) < 2:
            continue
        lookup = _NAME_ALIASES.get(cand.lower(), cand)
        for a in catalog_store.search_by_name(lookup):
            if a.id not in seen:
                seen.add(a.id)
                ids.append(a.id)
    return ids


def _build_facets(messages: list[Message]) -> tuple[list[str], list[str], list[int]]:
    """Decompose the conversation into retrieval facets.

    Returns (semantic_facets, lexical_terms, named_ids).

    A multi-skill request ("Java, Spring, SQL, AWS, Docker") becomes one
    semantic facet per skill, so every skill gets its own slice of the
    candidate pool instead of being averaged away by a single query vector.
    Because this runs over the FULL user history, a later "add personality
    tests" turn simply adds a facet — refinement augments instead of
    restarting.
    """
    user_msgs = [m.content for m in messages if m.role == "user"]
    if not user_msgs:
        return [], [], []

    joined = " ".join(user_msgs)
    latest = user_msgs[-1]
    recent = ". ".join(user_msgs[-4:])

    # Per-skill semantic facets (also serve as lexical match terms).
    skill_terms = retriever.extract_skill_terms(joined)

    # Order: latest message (most weight), recent context, then each skill.
    facets, seen = [], set()
    for f in [latest, recent] + skill_terms:
        key = f.strip().lower()
        if f.strip() and key not in seen:
            seen.add(key)
            facets.append(f)

    # Pin named assessments only on comparison turns (keeps RECOMMEND clean).
    named_ids = []
    if any(m in latest.lower() for m in _COMPARE_MARKERS):
        named_ids = _detect_named_assessments(latest)

    return facets, skill_terms, named_ids


_FACET_PLAN_INSTRUCTION = (
    "You expand a hiring conversation into catalog search phrases for an "
    "assessment recommender. Return ONLY a JSON array of 4-9 short phrases "
    "(no prose). Include both the explicit skills/role AND inferred needs:\n"
    "- list each named technical skill separately (e.g. \"Core Java\", "
    "\"Docker\", \"SQL\");\n"
    "- map roles to typical tools (admin/clerical -> \"Microsoft Word\", "
    "\"Microsoft Excel\"; healthcare -> \"medical terminology\"; "
    "safety-critical -> \"dependability and safety\");\n"
    "- for almost any professional/graduate/leadership hire, also include "
    "\"personality questionnaire OPQ\" and \"cognitive ability reasoning\".\n"
    "Do not invent product names; use generic skill/category phrases. "
    "Output a JSON array of strings only."
)


def _plan_facets_with_llm(messages: list[Message]) -> list[str]:
    """Use the LLM as a query planner to surface inferred search facets.

    Returns [] (caller falls back to rule-based facets) on any failure.
    """
    if not _llm_available():
        return []
    convo = "\n".join(
        f"{m.role}: {m.content}" for m in messages if m.role == "user"
    )
    if not convo.strip():
        return []
    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=_FACET_PLAN_INSTRUCTION,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )
        resp = model.generate_content(convo)
        arr = json.loads(resp.text)
        if isinstance(arr, list):
            facets = [str(x).strip() for x in arr if str(x).strip()]
            logger.info(f"LLM planner facets: {facets}")
            return facets[:9]
    except Exception as e:
        logger.warning(f"Facet planning failed, using rule-based facets: {e}")
    return []


def _get_relevant_assessments(
    messages: list[Message], pool_size: int = 40
) -> list[Assessment]:
    """Retrieve a unioned candidate pool from the full conversation context."""
    facets, lexical_terms, named_ids = _build_facets(messages)
    if not facets:
        return []

    # Augment with LLM-planned facets (inferred needs the literal words miss,
    # e.g. 'admin' -> Microsoft Word, 'healthcare' -> medical terminology,
    # any professional role -> personality + cognitive). Falls back silently
    # to the rule-based facets if the planner is unavailable or errors.
    for f in _plan_facets_with_llm(messages):
        if f.strip().lower() not in {x.lower() for x in facets}:
            facets.append(f)

    boost_types = _extract_test_type_boost(messages)
    results = retriever.retrieve_candidates(
        facets=facets,
        lexical_terms=lexical_terms,
        boost_types=boost_types,
        named_ids=named_ids,
        top_k=pool_size,
    )
    logger.info(
        f"Retrieved {len(results)} candidates "
        f"({len(facets)} facets, {len(lexical_terms)} skill terms, "
        f"{len(named_ids)} named) for: {facets[0][:60]}..."
    )
    return [a for a, _ in results]


def _parse_llm_response(raw_text: str) -> dict:
    """Parse the LLM's JSON response, handling common formatting issues."""
    text = raw_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return a safe default
    logger.warning(f"Could not parse LLM response as JSON: {text[:200]}")
    return {
        "action": "CLARIFY",
        "reply": text if len(text) < 500 else "Could you tell me more about the role you're hiring for?",
        "recommendations": [],
        "end_of_conversation": False,
    }


def _validate_recommendations(recommendations: list[dict]) -> list[Recommendation]:
    """Validate that all recommendations exist in the catalog."""
    validated = []
    for rec in recommendations:
        name = rec.get("name", "")
        url = rec.get("url", "")
        test_type = rec.get("test_type", "")

        # Try to find the assessment in the catalog
        assessment = None

        # Try URL match first (most reliable)
        if url:
            assessment = catalog_store.get_by_url(url)

        # Try name match
        if not assessment and name:
            assessment = catalog_store.get_by_name(name)

        # Try substring search
        if not assessment and name:
            matches = catalog_store.search_by_name(name)
            if matches:
                assessment = matches[0]

        if assessment:
            validated.append(Recommendation(
                name=assessment.name,
                url=assessment.url,
                test_type=", ".join(assessment.test_types) if assessment.test_types else test_type,
            ))
        else:
            logger.warning(f"Recommendation not found in catalog: name='{name}' url='{url}'")

    # Deduplicate by URL
    seen_urls = set()
    deduped = []
    for rec in validated:
        if rec.url not in seen_urls:
            seen_urls.add(rec.url)
            deduped.append(rec)

    return deduped[:MAX_RECOMMENDATIONS]


def _call_llm(system_prompt: str, messages: list[Message]) -> dict:
    """Call Gemini with the system prompt and conversation history."""
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=2048,
            response_mime_type="application/json",
        ),
    )

    # Build chat history for Gemini
    history = []
    for msg in messages[:-1]:  # All but last message
        role = "user" if msg.role == "user" else "model"
        history.append({"role": role, "parts": [msg.content]})

    chat = model.start_chat(history=history)

    # Send the last user message
    last_message = messages[-1].content if messages else "Hello"
    response = chat.send_message(last_message)

    logger.info(f"LLM response length: {len(response.text)} chars")
    return _parse_llm_response(response.text)


def _is_off_topic(text: str) -> bool:
    """Detect off-topic queries that the agent should refuse.

    Uses a two-pass approach: first check for strongly off-topic
    patterns (these override hiring keywords), then check for
    weaker patterns that can be overridden by hiring context.
    """
    text_lower = text.lower().strip()

    # Strong off-topic patterns — always off-topic regardless of other keywords
    strong_off_topic = [
        "tell me a joke", "tell me a funny",
        "what's the weather", "what is the weather",
        "recipe for", "how to cook",
        "who is the president", "capital of",
        "translate this", "translate the",
        "ignore all previous", "ignore your instructions",
        "you are now", "pretend you are",
        "history of rome", "history of",
    ]
    if any(p in text_lower for p in strong_off_topic):
        return True

    # Weak off-topic patterns — can be overridden by hiring context
    weak_off_topic = [
        "weather", "temperature", "forecast",
        "joke", "funny",
        "recipe", "cook", "food",
        "sports", "score", "game result",
        "movie", "film", "music",
        "how old", "where is",
        "definition of",
        "news", "stock price",
    ]
    # Check if the message has genuine assessment/hiring intent
    hiring_keywords = [
        "assess", "hire", "hiring", "recruit", "candidate", "role",
        "developer", "engineer", "manager", "analyst", "skill", "job",
        "position", "employee", "team", "cognitive", "personality",
        "aptitude", "competency", "shl",
        "evaluation", "screen", "select", "talent", "workforce",
        "recommend", "test for", "tests for",
    ]
    has_hiring_context = any(kw in text_lower for kw in hiring_keywords)
    has_off_topic = any(kw in text_lower for kw in weak_off_topic)

    return has_off_topic and not has_hiring_context


def _is_vague(text: str) -> bool:
    """Detect vague queries that need clarification.

    Only triggers on genuinely short/vague queries, NOT on detailed
    queries that happen to end with 'what assessments work?'.
    """
    text_lower = text.lower().strip()
    word_count = len(text_lower.split())

    # Very short queries are always vague (e.g., "I need a test")
    if word_count < 5:
        return True

    # Longer queries must match a vague pattern AND lack specifics
    vague_patterns = [
        "i need an assessment",
        "i need a test",
        "help me find",
        "what do you have",
        "recommend something",
        "show me assessments",
    ]
    matches_vague = any(p in text_lower for p in vague_patterns)

    # But only if the query doesn't have specific context
    specific_keywords = [
        "java", "python", "developer", "engineer", "manager", "analyst",
        "sales", "customer", "leadership", "cognitive", "numerical",
        "verbal", "personality", "coding", "programming", "data",
        "finance", "accounting", "marketing", "it ", "support",
        "graduate", "executive", "senior", "junior",
        "contact", "admin", "healthcare", "safety", "operator",
        "bilingual", "spanish", "excel", "word", "plant",
        "rust", "networking", "screening", "trainee",
    ]
    has_specifics = any(kw in text_lower for kw in specific_keywords)

    return matches_vague and not has_specifics and word_count < 12


def _fallback_recommend(messages: list[Message]) -> ChatResponse:
    """Fallback: use retriever-only recommendations when LLM is unavailable."""
    logger.info("Using fallback recommendation (no LLM)")

    # Get the latest user message
    user_text = " ".join(m.content for m in messages if m.role == "user")
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")

    # Check for off-topic
    if _is_off_topic(last_user):
        return ChatResponse(
            reply="I'm the SHL Assessment Advisor, and I can only help with finding SHL assessments for hiring and talent evaluation. Could you tell me about a role you're looking to hire for?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Check for vague queries (only on first turn)
    assistant_msgs = [m for m in messages if m.role == "assistant"]
    if not assistant_msgs and _is_vague(last_user):
        return ChatResponse(
            reply="I'd be happy to help! To recommend the best SHL assessments, could you tell me:\n- What role or position are you hiring for?\n- Are there specific skills or qualities you'd like to assess (e.g., technical skills, personality, cognitive ability)?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Retrieve and recommend
    relevant = _get_relevant_assessments(messages)

    if not relevant:
        return ChatResponse(
            reply="I'd be happy to help you find the right SHL assessments. Could you tell me about the role you're hiring for and what skills you'd like to evaluate?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Build recommendations from top results
    recs = []
    for a in relevant[:10]:
        recs.append(Recommendation(
            name=a.name,
            url=a.url,
            test_type=", ".join(a.test_types),
        ))

    return ChatResponse(
        reply=f"Based on your requirements, here are the most relevant SHL assessments I'd recommend. Each assessment is linked to its catalog page for more details:",
        recommendations=recs,
        end_of_conversation=False,
    )


def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Main entry point for processing a chat request.

    This is the core agent function that:
    1. Analyzes the conversation to retrieve relevant assessments
    2. Sends the conversation + retrieved context to the LLM
    3. Validates and returns the response in the correct schema
    """
    messages = request.messages

    # Handle empty messages
    if not messages:
        return ChatResponse(
            reply="Hello! I'm the SHL Assessment Advisor. I can help you find the right SHL assessments for your hiring needs. What role are you looking to hire for?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Check if LLM is available
    if not _llm_available():
        logger.warning("LLM unavailable — using retriever-only fallback mode")
        return _fallback_recommend(messages)

    try:
        # Step 1: Retrieve a candidate pool from the conversation context.
        # The LLM re-ranks/selects, so we give it a generous pool (recall
        # ceiling rises sharply with pool size) but cap the prompt to keep
        # it focused and well under the 30s call budget.
        relevant_assessments = _get_relevant_assessments(messages, pool_size=40)

        # Step 2: Build the system prompt with retrieved catalog context
        catalog_context = format_catalog_for_context(relevant_assessments[:30])
        system_prompt = build_system_prompt(catalog_context)

        # Step 3: Call the LLM
        llm_response = _call_llm(system_prompt, messages)

        # Step 4: Validate recommendations against catalog
        raw_recommendations = llm_response.get("recommendations", [])
        validated_recommendations = _validate_recommendations(raw_recommendations)

        # Step 5: Construct schema-compliant response
        reply = llm_response.get(
            "reply",
            "I apologize, but I encountered an issue. Could you please rephrase your request?"
        )
        end_of_conversation = llm_response.get("end_of_conversation", False)

        return ChatResponse(
            reply=reply,
            recommendations=validated_recommendations,
            end_of_conversation=end_of_conversation,
        )

    except Exception as e:
        logger.error(f"Agent error: {e}")
        traceback.print_exc()

        # Try fallback recommendation before giving up
        try:
            return _fallback_recommend(messages)
        except Exception:
            pass

        # Absolute last resort — never crash, always return valid schema
        return ChatResponse(
            reply="I'd be happy to help you find the right SHL assessments. Could you tell me about the role you're hiring for and what kind of skills or qualities you want to assess?",
            recommendations=[],
            end_of_conversation=False,
        )
