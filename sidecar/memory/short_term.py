"""
Short-Term Memory — SQLite Session Store
Phase 6.1: Persistent conversation history across sessions

Uses plain SQLite (encrypted SQLite via sqlcipher deferred to production).
Stores conversation threads with sliding-window retrieval.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Optional

logger = logging.getLogger("sylph.memory.short_term")

DEFAULT_DB_DIR = os.path.expanduser("~/.sylph/memory")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "sessions.db")


class ShortTermMemory:
    """
    SQLite-backed conversation memory.
    Stores messages per thread_id with timestamps.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    def initialize(self) -> None:
        """Create the database and tables if they don't exist."""
        if self._initialized:
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                processed INTEGER DEFAULT 0,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_thread
                ON messages(thread_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_messages_unprocessed
                ON messages(processed) WHERE processed = 0;
        """)
        self._conn.commit()
        self._initialized = True
        logger.info("Short-term memory initialized at %s", self._db_path)

    def _ensure_init(self) -> sqlite3.Connection:
        if not self._initialized or self._conn is None:
            self.initialize()
        assert self._conn is not None
        return self._conn

    def ensure_thread(self, thread_id: str) -> None:
        """Create a thread if it doesn't exist."""
        conn = self._ensure_init()
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO threads (thread_id, created_at, updated_at) VALUES (?, ?, ?)",
            (thread_id, now, now),
        )
        conn.commit()

    def add_message(self, thread_id: str, role: str, content: str) -> int:
        """
        Add a message to a thread.

        Returns:
            The message ID
        """
        conn = self._ensure_init()
        now = datetime.utcnow().isoformat()

        self.ensure_thread(thread_id)

        cursor = conn.execute(
            "INSERT INTO messages (thread_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (thread_id, role, content, now),
        )
        conn.execute(
            "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
            (now, thread_id),
        )
        conn.commit()

        msg_id = cursor.lastrowid or 0
        logger.debug("Message added: thread=%s, role=%s, id=%d", thread_id, role, msg_id)
        return msg_id

    def get_recent_messages(
        self, thread_id: str, limit: int = 20
    ) -> list[dict]:
        """
        Get the most recent messages from a thread (sliding window).

        Returns:
            List of {"role": str, "content": str, "timestamp": str}
        """
        conn = self._ensure_init()
        rows = conn.execute(
            """
            SELECT role, content, timestamp FROM messages
            WHERE thread_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (thread_id, limit),
        ).fetchall()

        # Reverse to get chronological order
        messages = [
            {"role": row[0], "content": row[1], "timestamp": row[2]}
            for row in reversed(rows)
        ]
        return messages

    def get_unprocessed_messages(self, limit: int = 50) -> list[dict]:
        """
        Get unprocessed messages for the sync worker (Phase 6.3).

        Returns:
            List of {"id": int, "thread_id": str, "role": str, "content": str, "timestamp": str}
        """
        conn = self._ensure_init()
        rows = conn.execute(
            """
            SELECT id, thread_id, role, content, timestamp FROM messages
            WHERE processed = 0
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            {
                "id": row[0],
                "thread_id": row[1],
                "role": row[2],
                "content": row[3],
                "timestamp": row[4],
            }
            for row in rows
        ]

    def mark_processed(self, message_ids: list[int]) -> None:
        """Mark messages as processed by the sync worker."""
        if not message_ids:
            return
        conn = self._ensure_init()
        placeholders = ",".join("?" for _ in message_ids)
        conn.execute(
            f"UPDATE messages SET processed = 1 WHERE id IN ({placeholders})",
            message_ids,
        )
        conn.commit()
        logger.debug("Marked %d messages as processed", len(message_ids))

    def list_threads(self, limit: int = 20) -> list[dict]:
        """List recent threads."""
        conn = self._ensure_init()
        rows = conn.execute(
            "SELECT thread_id, created_at, updated_at FROM threads ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"thread_id": row[0], "created_at": row[1], "updated_at": row[2]}
            for row in rows
        ]

    def get_all_messages(self, limit: int = 100) -> list[dict]:
        """Get all messages across threads (for settings memory viewer)."""
        conn = self._ensure_init()
        rows = conn.execute(
            "SELECT id, thread_id, role, content, timestamp FROM messages ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "thread_id": row[1],
                "role": row[2],
                "content": row[3],
                "timestamp": row[4],
            }
            for row in rows
        ]

    def delete_message(self, msg_id: int) -> None:
        """Delete a single message."""
        conn = self._ensure_init()
        conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
        conn.commit()
        logger.info("Deleted short-term message with ID %d", msg_id)

    def clear_all(self) -> None:
        """Clear all messages and threads."""
        conn = self._ensure_init()
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM threads")
        conn.commit()
        logger.info("Cleared all short-term memory tables")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False

