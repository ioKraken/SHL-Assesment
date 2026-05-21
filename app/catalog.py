"""Catalog loader — loads the scraped SHL catalog into memory."""

import json
import os
import logging
from typing import Optional
from app.config import CATALOG_PATH, TEST_TYPE_MAP

logger = logging.getLogger("shl-agent.catalog")


class Assessment:
    """Represents a single SHL assessment from the catalog."""

    def __init__(self, data: dict):
        self.id: int = data["id"]
        self.name: str = data["name"]
        self.url: str = data["url"]
        self.remote_testing: bool = data.get("remote_testing", False)
        self.adaptive_irt: bool = data.get("adaptive_irt", False)
        self.test_types: list[str] = data.get("test_types", [])
        self.description: str = data.get("description", "")
        self.duration_minutes: Optional[int] = data.get("duration_minutes")
        self.languages: list[str] = data.get("languages", [])
        self.job_levels: list[str] = data.get("job_levels", [])

    @property
    def test_type_str(self) -> str:
        """Human-readable test type string."""
        return ", ".join(self.test_types) if self.test_types else "Unknown"

    @property
    def test_type_descriptions(self) -> str:
        """Full descriptions of test types."""
        return "; ".join(
            TEST_TYPE_MAP.get(t, t) for t in self.test_types
        )

    def to_embedding_text(self) -> str:
        """Create a rich text document for embedding.

        Includes job levels, duration, and languages so that
        queries like 'entry-level', 'Spanish', '15 minutes'
        produce good semantic matches.
        """
        parts = [
            f"Name: {self.name}",
            f"Test Types: {self.test_type_descriptions}",
        ]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.job_levels:
            parts.append(f"Job Levels: {', '.join(self.job_levels)}")
        if self.duration_minutes:
            parts.append(f"Duration: {self.duration_minutes} minutes")
        if self.languages:
            parts.append(f"Languages: {', '.join(self.languages[:5])}")
        if self.remote_testing:
            parts.append("Supports remote/online testing")
        if self.adaptive_irt:
            parts.append("Adaptive/IRT scoring available")
        return "\n".join(parts)

    def to_catalog_entry(self) -> str:
        """Format for inclusion in LLM context."""
        parts = [f"- **{self.name}** ({self.test_type_str})"]
        if self.description:
            parts.append(f"  {self.description}")
        features = []
        if self.remote_testing:
            features.append("Remote")
        if self.adaptive_irt:
            features.append("Adaptive/IRT")
        if features:
            parts.append(f"  Features: {', '.join(features)}")
        parts.append(f"  URL: {self.url}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "remote_testing": self.remote_testing,
            "adaptive_irt": self.adaptive_irt,
            "test_types": self.test_types,
            "description": self.description,
        }


class CatalogStore:
    """In-memory store for the SHL assessment catalog."""

    def __init__(self):
        self.assessments: list[Assessment] = []
        self._by_id: dict[int, Assessment] = {}
        self._by_name: dict[str, Assessment] = {}
        self._by_url: dict[str, Assessment] = {}

    def load(self, path: str = CATALOG_PATH):
        """Load catalog from JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Catalog not found: {path}")

        with open(path) as f:
            data = json.load(f)

        self.assessments = [Assessment(item) for item in data]
        self._by_id = {a.id: a for a in self.assessments}
        self._by_name = {a.name.lower(): a for a in self.assessments}
        self._by_url = {a.url: a for a in self.assessments}

        logger.info(f"Loaded {len(self.assessments)} assessments from catalog")

    def get_by_id(self, id: int) -> Optional[Assessment]:
        return self._by_id.get(id)

    def get_by_name(self, name: str) -> Optional[Assessment]:
        """Case-insensitive name lookup."""
        return self._by_name.get(name.lower().strip())

    def search_by_name(self, query: str) -> list[Assessment]:
        """Fuzzy substring search on name. Returns up to 5 matches."""
        q = query.lower().strip()
        matches = [a for a in self.assessments if q in a.name.lower()]
        return matches[:5]

    def get_by_url(self, url: str) -> Optional[Assessment]:
        """Lookup by URL, handling trailing slash differences."""
        result = self._by_url.get(url)
        if not result and not url.endswith("/"):
            result = self._by_url.get(url + "/")
        if not result and url.endswith("/"):
            result = self._by_url.get(url.rstrip("/"))
        return result

    def get_by_test_type(self, test_type: str) -> list[Assessment]:
        """Filter by test type code."""
        return [a for a in self.assessments if test_type in a.test_types]

    def validate_url(self, url: str) -> bool:
        """Check if a URL exists in the catalog."""
        return self.get_by_url(url) is not None

    def validate_name(self, name: str) -> bool:
        """Check if a name exists in the catalog (case-insensitive)."""
        return name.lower().strip() in self._by_name


# Global catalog instance
catalog_store = CatalogStore()
