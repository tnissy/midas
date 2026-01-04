"""Configuration for Midas."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RAINDROP_API_TOKEN = os.getenv("RAINDROP_API_TOKEN")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NEWS_DIR = DATA_DIR / "news"

# Ensure directories exist
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# LLM Settings
LLM_MODEL = "gemini-3-flash-preview"
LLM_MAX_TOKENS = 4096


def extract_llm_text(content) -> str:
    """Extract text from LLM response content.

    Gemini 3 returns a list of dicts with 'text' key,
    while older models return a string directly.
    """
    if isinstance(content, list):
        if content and isinstance(content[0], dict):
            return content[0].get("text", "")
        return ""
    return content if isinstance(content, str) else ""
