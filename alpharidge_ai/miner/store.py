"""Local SQLite state: pages already read, claims already sent, their outcomes and their
signed receipts."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

from alpharidge_ai.schema import Claim


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
CREATE TABLE IF NOT EXISTS receipt (
    claim_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    hash TEXT NOT NULL,
    body TEXT NOT NULL
);
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

    def get(self, claim_id: str) -> Optional[dict]:
        with self._lock:
            row = self._db.execute("SELECT * FROM claim WHERE claim_id = ?",
                                   (str(claim_id),)).fetchone()
        return dict(row) if row else None

    def keep_receipt(self, receipt: dict) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO receipt (claim_id, seq, hash, body) VALUES (?, ?, ?, ?)",
                (str(receipt["claim_id"]), int(receipt["seq"]), str(receipt["hash"]),
                 json.dumps(receipt, sort_keys=True)))

    def receipts(self, limit: int = 1000) -> List[dict]:
        with self._lock:
            rows = self._db.execute("SELECT body FROM receipt ORDER BY seq LIMIT ?",
                                    (limit,)).fetchall()
        return [json.loads(r["body"]) for r in rows]
