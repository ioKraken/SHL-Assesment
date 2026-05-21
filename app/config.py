"""Configuration for the SHL Assessment Recommender Agent."""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("shl-agent")

# ─── LLM Configuration ─────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ─── Embedding Configuration ───────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# ─── Agent Configuration ───────────────────────────────────────
MAX_RECOMMENDATIONS = 10
MIN_RECOMMENDATIONS = 1
TOP_K_RETRIEVAL = 15   # Retrieve enough for filtering, but don't bloat LLM context
MAX_TURNS = 8

# ─── Server ─────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", "8000"))

# ─── Test type descriptions ────────────────────────────────────
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude (cognitive, numerical, verbal reasoning)",
    "B": "Behavioral (situational judgment, behavioral assessments)",
    "C": "Competency (competency-based assessments)",
    "D": "Development (development-focused tools, 360 feedback)",
    "E": "Experience (biodata, experience-based assessments)",
    "K": "Knowledge & Skills (technical knowledge, coding tests)",
    "P": "Personality (OPQ, personality questionnaires, motivation)",
    "S": "Simulation (work simulations, inbox exercises)",
}


def validate_config():
    """Validate critical configuration at startup."""
    issues = []
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        issues.append("GEMINI_API_KEY is not set. Set it in .env or as an environment variable.")
    if not os.path.exists(CATALOG_PATH):
        issues.append(f"Catalog file not found at {CATALOG_PATH}")
    return issues
