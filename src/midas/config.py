"""Configuration for Midas."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NEWS_DIR = DATA_DIR / "news"

# Ensure directories exist
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# LLM Settings
LLM_MODEL = "gemini-2.5-flash"
LLM_MAX_TOKENS = 4096
