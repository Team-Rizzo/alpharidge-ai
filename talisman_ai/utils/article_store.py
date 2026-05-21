from enum import Enum
import time
import json
from pathlib import Path
from typing import Optional, Dict, List

import bittensor as bt
from pydantic import BaseModel

from talisman_ai import config
from talisman_ai.utils.api_models import NewsArticleForScoring


class ArticleStatus(Enum):
    UNPROCESSED = "Unprocessed"
    PROCESSING = "Processing"
    PROCESSED = "Processed"


class ArticleStoreItem(BaseModel):
    article: NewsArticleForScoring
    status: ArticleStatus
    start_time: Optional[float] = None
    hotkey: Optional[str] = None
    # Idempotency helpers
    submitted_to_api: bool = False
    rewarded: bool = False


class ArticleStore:
    def __init__(self):
        # key: article_id, value: ArticleStoreItem
        self._articles: Dict[str, ArticleStoreItem] = {}

    def add_article(
        self,
        article: NewsArticleForScoring,
        article_id: Optional[str] = None,
        hotkey: Optional[str] = None,
        set_as_processing: bool = False,
        overwrite: bool = True,
    ):
        """
        Adds a news article to the store as Unprocessed.
        If article_id is provided, uses it as the key; else uses article.id.

        Args:
            article: NewsArticleForScoring object to store
            article_id: Optional article ID. If not provided, uses article.id
            hotkey: Optional miner hotkey processing this article
            set_as_processing: If True, set initial status to Processing instead of Unprocessed
            overwrite: If True, overwrite an existing entry. Defaults to True.
        """
        if article_id is None:
            article_id = str(article.id)
        # Normalize keys so persistence round-trips correctly (JSON object keys are strings).
        article_id = str(article_id)
        # Ensure article is a NewsArticleForScoring instance
        if not isinstance(article, NewsArticleForScoring):
            raise TypeError(f"article must be a NewsArticleForScoring instance, got {type(article)}")
        if (article_id in self._articles) and (not overwrite):
            # Preserve existing lifecycle/idempotency flags; optionally update article/hotkey if missing.
            existing = self._articles[article_id]
            if existing.article is None:
                existing.article = article
            # Only fill hotkey if not already set.
            if hotkey is not None and existing.hotkey is None:
                existing.hotkey = hotkey
            return
        self._articles[article_id] = ArticleStoreItem(
            article=article,
            status=ArticleStatus.PROCESSING if set_as_processing else ArticleStatus.UNPROCESSED,
            start_time=time.time() if set_as_processing else None,
            hotkey=hotkey,
            submitted_to_api=False,
            rewarded=False,
        )

    def update_article(self, article_id: str, article: NewsArticleForScoring):
        """Update the stored article object (e.g. attach miner analysis) without changing lifecycle flags."""
        article_id = str(article_id)
        if article_id not in self._articles:
            raise KeyError(f"Article ID {article_id} not found")
        self._articles[article_id].article = article

    def set_processing(self, article_id: str, hotkey: Optional[str] = None):
        """
        Sets the article as Processing and stores the current time as start_time.

        Args:
            article_id: ID of the article to set as processing
            hotkey: Optional miner hotkey processing this article
        """
        article_id = str(article_id)
        if article_id in self._articles:
            self._articles[article_id].status = ArticleStatus.PROCESSING
            self._articles[article_id].start_time = time.time()
            if hotkey is not None:
                self._articles[article_id].hotkey = hotkey
        else:
            raise KeyError(f"Article ID {article_id} not found")

    def set_processed(self, article_id: str):
        """
        Sets the article as Processed and clears start_time.
        """
        article_id = str(article_id)
        if article_id in self._articles:
            self._articles[article_id].status = ArticleStatus.PROCESSED
            self._articles[article_id].start_time = None
        else:
            raise KeyError(f"Article ID {article_id} not found")

    def reset_to_unprocessed(self, article_id: str):
        """
        Resets status to Unprocessed and clears start_time.
        Note: hotkey is preserved when resetting.
        """
        article_id = str(article_id)
        if article_id in self._articles:
            self._articles[article_id].status = ArticleStatus.UNPROCESSED
            self._articles[article_id].start_time = None
        else:
            raise KeyError(f"Article ID {article_id} not found")

    def get_article(self, article_id: str) -> NewsArticleForScoring:
        """
        Returns the stored article object.

        Returns:
            NewsArticleForScoring: The stored article object
        """
        article_id = str(article_id)
        if article_id in self._articles:
            return self._articles[article_id].article
        else:
            raise KeyError(f"Article ID {article_id} not found")

    def get_status(self, article_id: str) -> ArticleStatus:
        """
        Returns the current status of the article.
        """
        article_id = str(article_id)
        if article_id in self._articles:
            return self._articles[article_id].status
        else:
            raise KeyError(f"Article ID {article_id} not found")

    def get_hotkey(self, article_id: str) -> Optional[str]:
        """
        Returns the hotkey of the miner processing the article, if set.

        Returns:
            Optional[str]: The miner hotkey, or None if not set
        """
        article_id = str(article_id)
        if article_id in self._articles:
            return self._articles[article_id].hotkey
        else:
            raise KeyError(f"Article ID {article_id} not found")

    def get_all(self) -> Dict[str, ArticleStoreItem]:
        """
        Returns a copy of the full internal dict of all articles.
        """
        return dict(self._articles)

    def get_unprocessed_articles(self) -> List[ArticleStoreItem]:
        """
        Returns list of ArticleStoreItem that are in Unprocessed.
        """
        return [item for item in self._articles.values() if item.status == ArticleStatus.UNPROCESSED]

    def get_processing_articles(self) -> List[ArticleStoreItem]:
        """
        Returns list of articles that are in Processing.
        """
        return [item for item in self._articles.values() if item.status == ArticleStatus.PROCESSING]

    def get_processed_articles(self) -> List[ArticleStoreItem]:
        """
        Returns list of articles that are in Processed.
        """
        return [item for item in self._articles.values() if item.status == ArticleStatus.PROCESSED]

    def get_ready_to_submit(self) -> List[ArticleStoreItem]:
        """Return processed articles that have not yet been submitted to the API."""
        return [
            item for item in self._articles.values()
            if item.status == ArticleStatus.PROCESSED and not item.submitted_to_api
        ]

    def get_timeouts(self) -> List[ArticleStoreItem]:
        """
        Returns list of articles that are in Processing and have exceeded the scoring lease TTL.
        """
        now = time.time()
        timeout = float(getattr(config, "SCORING_LEASE_TTL_SECONDS", 900))
        result = []
        for item in self._articles.values():
            if (
                item.status == ArticleStatus.PROCESSING
                and item.start_time is not None
                and (now - item.start_time) > timeout
            ):
                result.append(item)
        return result

    def mark_submitted(self, article_id: str):
        """Mark a processed article as successfully submitted to the API."""
        article_id = str(article_id)
        if article_id not in self._articles:
            raise KeyError(f"Article ID {article_id} not found")
        self._articles[article_id].submitted_to_api = True

    def mark_rewarded(self, article_id: str):
        """Mark an article as having already contributed reward to a miner."""
        article_id = str(article_id)
        if article_id not in self._articles:
            raise KeyError(f"Article ID {article_id} not found")
        self._articles[article_id].rewarded = True

    def is_rewarded(self, article_id: str) -> bool:
        article_id = str(article_id)
        if article_id not in self._articles:
            return False
        return bool(self._articles[article_id].rewarded)

    def delete_article(self, article_id: str):
        """
        Deletes an article from the store.
        """
        article_id = str(article_id)
        if article_id in self._articles:
            del self._articles[article_id]
        else:
            raise KeyError(f"Article ID {article_id} not found")

    def prune_old_articles(self, max_age_seconds: float = 3600, max_articles: int = 1000):
        """
        Prune old articles to prevent unbounded memory growth.

        Removes:
        1. All submitted articles (submitted_to_api=True)
        2. Unprocessed articles older than max_age_seconds
        3. If still over max_articles, remove oldest articles

        Args:
            max_age_seconds: Maximum age for unprocessed articles (default: 1 hour)
            max_articles: Maximum number of articles to keep (default: 1000)
        """
        now = time.time()

        # First pass: remove submitted and old unprocessed articles
        article_ids_to_delete = []
        for article_id, item in self._articles.items():
            # Remove submitted articles
            if item.submitted_to_api:
                article_ids_to_delete.append(article_id)
                continue
            # Remove old unprocessed articles
            if item.status == ArticleStatus.UNPROCESSED:
                if item.start_time is not None and (now - item.start_time) > max_age_seconds:
                    article_ids_to_delete.append(article_id)

        for article_id in article_ids_to_delete:
            del self._articles[article_id]

        # Second pass: if still over limit, remove oldest articles
        if len(self._articles) > max_articles:
            # Sort by start_time (oldest first), keeping None at the end
            sorted_items = sorted(
                self._articles.items(),
                key=lambda x: x[1].start_time if x[1].start_time is not None else float('inf')
            )
            # Keep only the newest max_articles
            to_remove = len(self._articles) - max_articles
            for article_id, _ in sorted_items[:to_remove]:
                del self._articles[article_id]

    def save_to_file(self, file_path: Optional[str] = None):
        """
        Saves the article store to a JSON file.

        Args:
            file_path: Path to the file. Defaults to config.ARTICLE_STORE_LOCATION
        """
        if file_path is None:
            file_path = getattr(config, 'ARTICLE_STORE_LOCATION', 'article_store.json')

        file_path = Path(file_path)
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for serialization
        data = {
            "articles": {}
        }

        for article_id, item in self._articles.items():
            data["articles"][article_id] = item.model_dump(mode='json')

        # Write to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        bt.logging.debug(f"Article store saved to {file_path} ({len(self._articles)} articles)")

    def load_from_file(self, file_path: Optional[str] = None):
        """
        Loads the article store from a JSON file.

        Args:
            file_path: Path to the file. Defaults to config.ARTICLE_STORE_LOCATION
        """
        if file_path is None:
            file_path = getattr(config, 'ARTICLE_STORE_LOCATION', 'article_store.json')

        file_path = Path(file_path)

        # If file doesn't exist, start with empty store
        if not file_path.exists():
            self._articles = {}
            bt.logging.debug(f"No article store file found at {file_path}, starting empty")
            return

        # Read from file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Clear existing articles
        self._articles = {}

        # Deserialize articles
        for article_id, item_data in data.get("articles", {}).items():
            article_id = str(article_id)
            # Reconstruct NewsArticleForScoring from dict
            article = NewsArticleForScoring.model_validate(item_data["article"])
            # Reconstruct status enum
            status = ArticleStatus(item_data["status"])
            start_time = item_data.get("start_time")
            hotkey = item_data.get("hotkey")
            submitted_to_api = bool(item_data.get("submitted_to_api", False))
            rewarded = bool(item_data.get("rewarded", False))

            self._articles[article_id] = ArticleStoreItem(
                article=article,
                status=status,
                start_time=start_time,
                hotkey=hotkey,
                submitted_to_api=submitted_to_api,
                rewarded=rewarded,
            )

        bt.logging.debug(f"Article store loaded from {file_path} ({len(self._articles)} articles)")
