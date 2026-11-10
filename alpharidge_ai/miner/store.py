"""Local SQLite state: pages already read, claims already sent and their outcomes."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

from alpharidge_ai.schema import Claim

FINAL = frozenset({"settled", "clawed_back", "rejected"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS page (
    url TEXT PRIMARY KEY,
    seen_at REAL NOT NULL,
    outcome TEXT
);
CREATE TABLE IF NOT EXISTS claim (
    body_hash TEXT PRIMARY KEY,
    claim_id TEXT,
    kind TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_id TEXT,
    source_url TEXT NOT NULL,
    status TEXT,
    reason TEXT,
    provisional REAL DEFAULT 0,
    settled REAL DEFAULT 0,
    clawed_back REAL DEFAULT 0,
    sent_at REAL NOT NULL,
    checked_at REAL
);
CREATE INDEX IF NOT EXISTS claim_open ON claim (status, checked_at);
"""


class MinerStore:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def seen(self, url: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM page WHERE url = ?", (url,)).fetchone() is not None

    def mark_seen(self, url: str, outcome: str = "") -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO page (url, seen_at, outcome) VALUES (?, ?, ?)",
                             (url, time.time(), outcome))

    def sent(self, claim: Claim) -> bool:
        """Whether this exact claim, or one for the same slot of the same page, went out."""
        with self._lock:
            if self._db.execute("SELECT 1 FROM claim WHERE body_hash = ?",
                                (claim.claim_hash(),)).fetchone():
                return True
            return self._db.execute(
                "SELECT 1 FROM claim WHERE source_url = ? AND kind = ? AND event_type = ? "
                "AND COALESCE(event_id, '') = ?",
                (claim.source_url, claim.kind.value, claim.event_type.value,
                 claim.event_id or "")).fetchone() is not None

    def record(self, claim: Claim, response: dict) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO claim (body_hash, claim_id, kind, event_type, event_id, "
                "source_url, status, sent_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (claim.claim_hash(), str(response.get("id")) if response.get("id") else None,
                 claim.kind.value, claim.event_type.value, claim.event_id, claim.source_url,
                 response.get("status"), time.time()))

    def open_claims(self, limit: int = 50) -> List[str]:
        """Claim ids without a final outcome, least recently checked first."""
        marks = ",".join("?" * len(FINAL))
        with self._lock:
            rows = self._db.execute(
                f"SELECT claim_id FROM claim WHERE claim_id IS NOT NULL AND "
                f"(status IS NULL OR status NOT IN ({marks})) "
                f"ORDER BY COALESCE(checked_at, 0) LIMIT ?", (*sorted(FINAL), limit)).fetchall()
        return [r["claim_id"] for r in rows]

    def update(self, claim_id: str, status: dict) -> None:
        points = status.get("points") or {}
        with self._lock, self._db:
            self._db.execute(
                "UPDATE claim SET status = ?, reason = ?, provisional = ?, settled = ?, "
                "clawed_back = ?, checked_at = ? "
                "WHERE claim_id = ?",
                (status.get("status"), status.get("reason"), float(points.get("provisional", 0)),
                 float(points.get("settled", 0)), float(points.get("clawed_back", 0)),
                 time.time(), str(claim_id)))

    def get(self, claim_id: str) -> Optional[dict]:
        with self._lock:
            row = self._db.execute("SELECT * FROM claim WHERE claim_id = ?",
                                   (str(claim_id),)).fetchone()
        return dict(row) if row else None
