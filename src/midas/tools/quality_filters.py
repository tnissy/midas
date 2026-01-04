"""News quality filtering functions."""

import json
import re
from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from midas.config import GEMINI_API_KEY, LLM_MODEL, extract_llm_text
from midas.models import NewsItem, NewsCluster
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# LLM Client
# =============================================================================

_llm_client = None


def get_llm_client() -> ChatGoogleGenerativeAI:
    """Get or create LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            api_key=GEMINI_API_KEY,
            temperature=0.0
        )
    return _llm_client


# =============================================================================
# Author Extraction and Blacklist Check
# =============================================================================


def extract_author(news_item: NewsItem) -> str | None:
    """
    Extract author name from news content.

    Args:
        news_item: News item

    Returns:
        Author name or None
    """
    # Common patterns for author attribution
    patterns = [
        r"By ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",  # By John Doe
        r"(?:記者|執筆|著者)[：:]\s*([^\n]+)",  # Japanese: 記者: 山田太郎
        r"(?:Written by|Author)[：:]\s*([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
    ]

    content = news_item.content
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1).strip()

    return None


def load_author_blacklist(blacklist_path: Path | None = None) -> list[dict]:
    """
    Load author blacklist from JSON file.

    Args:
        blacklist_path: Path to blacklist file

    Returns:
        List of blacklisted authors
    """
    if blacklist_path is None:
        from midas.config import DATA_DIR
        blacklist_path = DATA_DIR / "blacklist" / "authors.json"

    if not blacklist_path.exists():
        return []

    try:
        with open(blacklist_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load blacklist: {e}")
        return []


def check_author_blacklist(news_item: NewsItem, author: str | None = None) -> bool:
    """
    Check if the author is blacklisted.

    Args:
        news_item: News item
        author: Author name (if already extracted)

    Returns:
        True if blacklisted, False otherwise
    """
    if author is None:
        author = extract_author(news_item)

    if author is None:
        return False

    blacklist = load_author_blacklist()
    blacklisted_names = {entry["name"].lower() for entry in blacklist}

    return author.lower() in blacklisted_names


# =============================================================================
# Duplicate News Clustering (TF-IDF based, no LLM)
# =============================================================================


def cluster_duplicate_news(news_items: list[NewsItem]) -> tuple[list[NewsCluster], dict[str, str]]:
    """
    Cluster duplicate news articles using TF-IDF similarity (no LLM).

    Args:
        news_items: List of news items to cluster

    Returns:
        Tuple of (clusters, item_id -> cluster_id mapping)
    """
    if len(news_items) <= 1:
        return [], {}

    logger.info(f"Clustering {len(news_items)} news items using TF-IDF...")

    # Prepare texts (title + content)
    texts = [f"{item.title} {item.content[:500]}" for item in news_items]
    ids = [item.id for item in news_items]

    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except Exception as e:
        logger.error(f"TF-IDF vectorization failed: {e}")
        return [], {}

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Find clusters (similarity > 0.85)
    SIMILARITY_THRESHOLD = 0.85
    clustered = set()
    clusters: list[NewsCluster] = []
    item_to_cluster: dict[str, str] = {}

    for i, id_i in enumerate(ids):
        if id_i in clustered:
            continue

        # Find similar items
        similar_indices = []
        for j, id_j in enumerate(ids):
            if i != j and similarity_matrix[i][j] > SIMILARITY_THRESHOLD:
                similar_indices.append(j)

        if similar_indices:
            # Create cluster
            cluster_id = f"cluster_{datetime.now().strftime('%Y%m%d')}_{len(clusters)+1:03d}"

            # Select representative (highest quality source)
            cluster_item_ids = [id_i] + [ids[j] for j in similar_indices]
            cluster_items = [item for item in news_items if item.id in cluster_item_ids]

            # Prioritize: government > major media > others
            source_priority = {
                "whitehouse": 100,
                "fda": 95,
                "sec": 90,
                "congress": 85,
                "bloomberg": 80,
                "reuters": 75,
                "wsj": 70,
            }

            representative = max(
                cluster_items,
                key=lambda x: source_priority.get(x.source.lower(), 0)
            )

            # Build similarity scores
            similarity_scores = {}
            for j in similar_indices:
                similarity_scores[ids[j]] = float(similarity_matrix[i][j])

            # Create cluster
            cluster = NewsCluster(
                cluster_id=cluster_id,
                representative_news_id=representative.id,
                duplicate_news_ids=[id for id in cluster_item_ids if id != representative.id],
                similarity_scores=similarity_scores,
                created_at=datetime.now()
            )

            clusters.append(cluster)

            # Mark as clustered
            for item_id in cluster_item_ids:
                clustered.add(item_id)
                item_to_cluster[item_id] = cluster_id

    logger.info(f"Created {len(clusters)} clusters from {len(news_items)} items")
    return clusters, item_to_cluster


# =============================================================================
# Title Translation - now moved to news_watcher_base as batch processing
# =============================================================================


def detect_language(text: str) -> str:
    """
    Detect language of text.

    Args:
        text: Text to detect

    Returns:
        Language code (e.g., 'en', 'ja')
    """
    try:
        return detect(text)
    except Exception:
        return "unknown"


# =============================================================================
# Advertisement Detection (LLM)
# =============================================================================


def detect_advertisement(news_item: NewsItem) -> tuple[bool, str]:
    """
    Detect if news item is an advertisement using LLM.

    Args:
        news_item: News item to check

    Returns:
        Tuple of (is_advertisement, reason)
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm_client()

    system_prompt = """あなたは広告・プロモーション判定の専門家です。
ニュース記事が広告またはプロモーションコンテンツかどうか判定してください。

判定基準:
- 製品の割引情報（例：「iPod が 3割引き」）
- アフィリエイトリンクを含む記事
- 企業のプレスリリースをそのまま掲載した記事
- 明確な購買誘導がある記事
- ネイティブ広告

必ず JSON 形式で回答してください: {"is_advertisement": true/false, "reason": "理由"}"""

    user_prompt = f"""タイトル: {news_item.title}
本文: {news_item.content[:1000]}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        text = extract_llm_text(response.content)

        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            is_ad = result.get("is_advertisement", False)
            reason = result.get("reason", "")
            return is_ad, reason
        else:
            logger.warning(f"Failed to parse LLM response for advertisement detection: {text}")
            return False, "Failed to parse"

    except Exception as e:
        logger.error(f"Advertisement detection failed: {e}")
        return False, f"Error: {e}"


# =============================================================================
# News Value Assessment (LLM)
# =============================================================================


def assess_news_value(news_item: NewsItem) -> tuple[int, str]:
    """
    Assess news value using LLM.

    Args:
        news_item: News item to assess

    Returns:
        Tuple of (value_score 0-10, reason)
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm_client()

    category_str = news_item.category.value if news_item.category else "unknown"

    system_prompt = """あなたは長期投資判断の専門家です。
ニュース記事が長期投資判断にどれだけ価値があるか評価してください。

評価基準（0-10点）:
- 10点: 業界を変える構造的変化（法規制、技術革新、市場再編など）
- 7-9点: 特定セクターに大きな影響（大型M&A、政策変更など）
- 4-6点: 中程度の影響（企業戦略変更、中規模提携など）
- 1-3点: 些細なニュース（人事異動、四半期決算、マイナーアップデートなど）
- 0点: 価値なし

必ず JSON 形式で回答してください: {"value_score": 0-10, "reason": "理由"}"""

    user_prompt = f"""タイトル: {news_item.title}
本文: {news_item.content[:1000]}
カテゴリ: {category_str}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        text = extract_llm_text(response.content)

        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            value_score = result.get("value_score", 5)
            reason = result.get("reason", "")
            return int(value_score), reason
        else:
            logger.warning(f"Failed to parse LLM response for value assessment: {text}")
            return 5, "Failed to parse"

    except Exception as e:
        logger.error(f"Value assessment failed: {e}")
        return 5, f"Error: {e}"


# =============================================================================
# Title Translation - Batch Processing (LLM)
# =============================================================================


def translate_titles_batch(news_items: list[NewsItem]) -> dict[str, str]:
    """
    Translate English titles to Japanese using batch processing.

    Args:
        news_items: List of news items

    Returns:
        Dictionary mapping news_item.id -> japanese_title
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm_client()

    # Detect English titles only
    english_items = []
    for item in news_items:
        lang = detect_language(item.title)
        if lang == "en":
            english_items.append(item)

    if not english_items:
        logger.info("No English titles to translate")
        return {}

    logger.info(f"Translating {len(english_items)} English titles to Japanese...")

    # Build batch prompt (max 50 items per batch to avoid token limits)
    BATCH_SIZE = 50
    translations = {}

    for i in range(0, len(english_items), BATCH_SIZE):
        batch = english_items[i:i+BATCH_SIZE]
        batch_titles = "\n".join([f"{j+1}. {item.title}" for j, item in enumerate(batch)])

        system_prompt = """あなたは英日翻訳の専門家です。
ニュースタイトルを英語から日本語に翻訳してください。
各タイトルは番号付きリストで提供されます。同じ番号順で日本語タイトルを返してください。

必ず JSON 形式で回答してください:
{
  "translations": [
    "1つ目の日本語タイトル",
    "2つ目の日本語タイトル",
    ...
  ]
}"""

        user_prompt = f"""以下の英語ニュースタイトルを翻訳してください:

{batch_titles}"""

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            text = extract_llm_text(response.content)

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                batch_translations = result.get("translations", [])

                # Map to news items
                for j, item in enumerate(batch):
                    if j < len(batch_translations):
                        translations[item.id] = batch_translations[j]
                    else:
                        logger.warning(f"Missing translation for item {j+1}")
            else:
                logger.warning(f"Failed to parse LLM response for batch translation: {text}")

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            continue

    logger.info(f"Translated {len(translations)} titles")
    return translations
