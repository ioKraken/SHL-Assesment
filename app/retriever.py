"""Hybrid retriever for SHL assessments.

Combines:
  1. Multi-query semantic search over a FAISS index (sentence-transformers).
  2. A lexical name index for exact skill-name matches (Java, Docker, HIPAA...).
  3. A soft test-type boost (never a hard filter, so we don't delete correct
     answers from the candidate pool).

The public entry point is `retrieve_candidates()`, which returns a unioned,
re-ranked pool of (Assessment, score) tuples for the agent / LLM to choose from.
"""

import os
import re
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, FAISS_INDEX_PATH, TOP_K_RETRIEVAL
from app.catalog import Assessment, catalog_store

logger = logging.getLogger("shl-agent.retriever")

# Generic words that carry no skill signal — excluded from the lexical index
# so that a query token like "test" or "new" doesn't match half the catalog.
_LEXICAL_STOPWORDS = {
    "the", "and", "for", "with", "new", "test", "tests", "level", "advanced",
    "intermediate", "basic", "report", "reports", "assessment", "assessments",
    "skills", "skill", "knowledge", "solution", "solutions", "general",
    "focus", "essentials", "essential", "interactive", "short", "form",
    "professional", "entry", "job", "role", "global", "core", "plus",
    "version", "online", "remote", "v1", "v2", "8", "ii", "iii",
}

# A skill token mapping to more than this many catalog items is too generic
# to be a useful lexical signal (e.g. "manager", "data").
_MAX_LEXICAL_DOCS_PER_TERM = 12


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, keeping +/#/. so 'c++', 'c#', '.net' survive."""
    return [t for t in re.findall(r"[a-z0-9.+#]+", text.lower()) if len(t) >= 2]


class AssessmentRetriever:
    """Hybrid semantic + lexical search over SHL assessments."""

    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[int] = []          # FAISS position -> assessment id
        self.lexical_index: dict[str, set[int]] = {}  # skill token -> assessment ids
        self.skill_vocab: set[str] = set()     # specific (non-generic) skill tokens
        self._loaded = False

    # ── lifecycle ────────────────────────────────────────────────
    def initialize(self):
        """Load or build the FAISS index, then build the lexical index."""
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        meta_file = os.path.join(FAISS_INDEX_PATH, "metadata.json")

        if os.path.exists(index_file) and os.path.exists(meta_file):
            logger.info("Loading existing FAISS index...")
            self.index = faiss.read_index(index_file)
            with open(meta_file) as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
        else:
            logger.info("Building FAISS index from catalog...")
            self._build_index()

        self._build_lexical_index()
        self._loaded = True

    def _build_index(self):
        """Build FAISS index from catalog assessments."""
        assessments = catalog_store.assessments
        if not assessments:
            raise ValueError("Catalog is empty — load catalog before building index")

        texts = [a.to_embedding_text() for a in assessments]
        self.metadata = [a.id for a in assessments]

        logger.info(f"Encoding {len(texts)} assessments...")
        embeddings = self.model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        faiss.write_index(self.index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))
        with open(os.path.join(FAISS_INDEX_PATH, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)

        logger.info(f"Built and saved index with {self.index.ntotal} vectors (dim={dim})")

    def _build_lexical_index(self):
        """Build token -> assessment-ids index from catalog NAMES only.

        Names carry the strongest exact-match signal (e.g. 'Core Java',
        'Docker', 'HIPAA'). Generic tokens are dropped via a stopword list
        and a max-docs-per-term cap so only specific skill tokens remain.
        """
        raw: dict[str, set[int]] = {}
        for a in catalog_store.assessments:
            for tok in set(_tokenize(a.name)):
                if tok in _LEXICAL_STOPWORDS:
                    continue
                raw.setdefault(tok, set()).add(a.id)

        # Keep only specific tokens (drop the over-broad ones).
        self.lexical_index = {
            tok: ids for tok, ids in raw.items()
            if len(ids) <= _MAX_LEXICAL_DOCS_PER_TERM
        }
        self.skill_vocab = set(self.lexical_index.keys())
        logger.info(
            f"Lexical index: {len(self.skill_vocab)} specific skill tokens "
            f"(dropped {len(raw) - len(self.skill_vocab)} generic ones)"
        )

    # ── helpers ──────────────────────────────────────────────────
    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.array(
            self.model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )

    def extract_skill_terms(self, text: str) -> list[str]:
        """Return specific skill tokens from `text` that exist in the catalog.

        These double as (a) lexical match terms and (b) per-facet semantic
        queries, which is what lets multi-skill JDs surface every skill.
        """
        seen, terms = set(), []
        for tok in _tokenize(text):
            if tok in self.skill_vocab and tok not in seen:
                seen.add(tok)
                terms.append(tok)
        return terms

    def lexical_search(self, terms: list[str]) -> dict[int, int]:
        """Map assessment id -> number of distinct skill terms it matches."""
        hits: dict[int, int] = {}
        for term in terms:
            for aid in self.lexical_index.get(term.lower(), ()):  # set or empty
                hits[aid] = hits.get(aid, 0) + 1
        return hits

    # ── main retrieval ───────────────────────────────────────────
    def retrieve_candidates(
        self,
        facets: list[str],
        lexical_terms: list[str] | None = None,
        boost_types: list[str] | None = None,
        named_ids: list[int] | None = None,
        top_k: int = 40,
        per_facet_k: int = 12,
    ) -> list[tuple[Assessment, float]]:
        """Union multi-query semantic + lexical results into one ranked pool.

        Scoring (all monotonic, same 0..~1.6 scale):
          - semantic: best cosine across all facet queries (0..1)
          - lexical : +0.5 if the assessment's NAME matched a skill term
                      (+0.1 per extra distinct term, capped)
          - type    : +0.05 soft boost if it has a hinted test type
          - named   : items explicitly named by the user are pinned to the top
        """
        if not self._loaded:
            raise RuntimeError("Retriever not initialized — call initialize() first")

        score: dict[int, float] = {}

        # 1. Multi-query semantic search (batch-encoded → one model call)
        facets = [f for f in (facets or []) if f and f.strip()]
        if facets:
            embs = self._encode(facets)
            D = min(per_facet_k, self.index.ntotal)
            sims, idxs = self.index.search(embs, D)
            for row_s, row_i in zip(sims, idxs):
                for s, i in zip(row_s, row_i):
                    if i < 0:
                        continue
                    aid = self.metadata[i]
                    score[aid] = max(score.get(aid, 0.0), float(s))

        # 2. Lexical exact-name boost. Items whose NAME contains a queried
        #    skill term get a strong bump (these are usually exactly what the
        #    user means), with a small extra bump for matching several terms.
        if lexical_terms:
            for aid, cnt in self.lexical_search(lexical_terms).items():
                score[aid] = score.get(aid, 0.0) + 0.5 + 0.1 * min(cnt - 1, 3)

        # 3. Soft test-type boost
        if boost_types:
            for aid in list(score.keys()):
                a = catalog_store.get_by_id(aid)
                if a and any(t in a.test_types for t in boost_types):
                    score[aid] += 0.05

        # 4. Pin user-named assessments to the very top (for COMPARE / refine)
        if named_ids:
            for aid in named_ids:
                score[aid] = max(score.get(aid, 0.0), 0.0) + 100.0

        ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = []
        for aid, sc in ranked:
            a = catalog_store.get_by_id(aid)
            if a is not None:
                out.append((a, sc))
        return out

    # ── backwards-compatible single-query helper ─────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        boost_types: list[str] | None = None,
    ) -> list[tuple[Assessment, float]]:
        """Single-query convenience wrapper around retrieve_candidates."""
        return self.retrieve_candidates(
            facets=[query], boost_types=boost_types, top_k=top_k
        )

    def retrieve_by_names(self, names: list[str]) -> list[Assessment]:
        """Resolve specific assessments by (fuzzy) name — used for comparisons."""
        results, seen = [], set()
        for name in names:
            a = catalog_store.get_by_name(name)
            if not a:
                matches = catalog_store.search_by_name(name)
                a = matches[0] if matches else None
            if a and a.id not in seen:
                seen.add(a.id)
                results.append(a)
        return results


# Global retriever instance
retriever = AssessmentRetriever()
