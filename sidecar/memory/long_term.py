"""
Long-Term Memory — ChromaDB Vector Store
Phase 6.2: Persistent fact storage with semantic retrieval

Uses ChromaDB in persistent mode with three collections:
- user_facts: Personal information about the user
- episodic: Notable conversation events and interactions
- world_knowledge: General knowledge and preferences

Embedding via ChromaDB's built-in default model (all-MiniLM-L6-v2).
"""

import logging
import os
from datetime import datetime
from typing import Optional
from uuid import uuid4

import chromadb
from chromadb.config import Settings

logger = logging.getLogger("sylph.memory.long_term")

# Suppress noisy ChromaDB telemetry errors (posthog API mismatch — harmless)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

DEFAULT_PERSIST_DIR = os.path.expanduser("~/.sylph/memory/longterm")

COLLECTIONS = ["user_facts", "episodic", "world_knowledge"]


class LongTermMemory:
    """
    ChromaDB-backed long-term memory with semantic retrieval.
    Stores facts, episodes, and knowledge as embedded vectors.
    """

    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        self._persist_dir = persist_dir
        self._client: Optional[chromadb.ClientAPI] = None
        self._collections: dict[str, chromadb.Collection] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize ChromaDB client and collections."""
        if self._initialized:
            return

        os.makedirs(self._persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or get collections
        for name in COLLECTIONS:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

        self._initialized = True
        counts = {name: col.count() for name, col in self._collections.items()}
        logger.info("Long-term memory initialized: %s", counts)

    def _ensure_init(self) -> None:
        if not self._initialized:
            self.initialize()

    def store_fact(
        self,
        text: str,
        category: str = "user_facts",
        confidence: float = 0.8,
        source_thread: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store a fact in long-term memory.

        Args:
            text: The fact text (e.g., "User's name is Akash")
            category: Collection name (user_facts, episodic, world_knowledge)
            confidence: Confidence score 0.0-1.0
            source_thread: Thread ID where this fact was extracted from
            metadata: Additional metadata

        Returns:
            The generated fact ID
        """
        self._ensure_init()

        if category not in self._collections:
            logger.warning("Unknown category '%s', defaulting to user_facts", category)
            category = "user_facts"

        fact_id = str(uuid4())
        fact_metadata = {
            "confidence": confidence,
            "source_thread": source_thread,
            "created_at": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        self._collections[category].add(
            documents=[text],
            metadatas=[fact_metadata],
            ids=[fact_id],
        )

        logger.info("Stored fact [%s]: '%s' (confidence=%.2f)", category, text[:60], confidence)
        return fact_id

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 6,
        min_similarity: float = 0.65,
        categories: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve facts relevant to a query using semantic search.

        Args:
            query: Search query text
            top_k: Max results per collection
            min_similarity: Minimum cosine similarity (0.0-1.0)
            categories: Collections to search (None = all)

        Returns:
            List of {"text": str, "category": str, "similarity": float, "metadata": dict}
            sorted by similarity descending
        """
        self._ensure_init()

        search_collections = categories or COLLECTIONS
        all_results: list[dict] = []

        for cat_name in search_collections:
            col = self._collections.get(cat_name)
            if col is None or col.count() == 0:
                continue

            # Query ChromaDB
            actual_k = min(top_k, col.count())
            results = col.query(
                query_texts=[query],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                # ChromaDB returns cosine distance; convert to similarity
                similarity = 1.0 - dist
                if similarity >= min_similarity:
                    all_results.append({
                        "text": doc,
                        "category": cat_name,
                        "similarity": round(similarity, 3),
                        "metadata": meta,
                    })

        # Sort by similarity descending
        all_results.sort(key=lambda x: x["similarity"], reverse=True)

        # Limit total results
        results = all_results[:top_k]
        logger.debug("Retrieved %d relevant facts for query: '%s'", len(results), query[:50])
        return results

    def deduplicate(self, text: str, category: str = "user_facts", threshold: float = 0.90) -> bool:
        """
        Check if a similar fact already exists (for the sync worker).

        Returns:
            True if a duplicate was found (skip insertion)
        """
        self._ensure_init()
        col = self._collections.get(category)
        if col is None or col.count() == 0:
            return False

        results = col.query(
            query_texts=[text],
            n_results=1,
            include=["distances"],
        )

        distances = results.get("distances", [[]])[0]
        if distances:
            similarity = 1.0 - distances[0]
            if similarity >= threshold:
                logger.debug("Duplicate found (sim=%.3f): '%s'", similarity, text[:50])
                return True
        return False

    def get_collection_stats(self) -> dict:
        """Get count of facts per collection."""
        self._ensure_init()
        return {name: col.count() for name, col in self._collections.items()}

    def get_all_facts(self, limit: int = 100) -> list[dict]:
        """Retrieve all long-term facts across collections (for settings memory viewer)."""
        self._ensure_init()
        all_facts = []
        for name, col in self._collections.items():
            if col.count() == 0:
                continue
            results = col.get(limit=limit, include=["documents", "metadatas"])
            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            for fid, doc, meta in zip(ids, documents, metadatas):
                all_facts.append({
                    "id": fid,
                    "text": doc,
                    "category": name,
                    "metadata": meta,
                })
        # Sort by creation time descending if available
        all_facts.sort(key=lambda x: x.get("metadata", {}).get("created_at", ""), reverse=True)
        return all_facts[:limit]

    def delete_fact(self, fact_id: str, category: str) -> None:
        """Delete a single fact by ID from a specific collection."""
        self._ensure_init()
        col = self._collections.get(category)
        if col:
            col.delete(ids=[fact_id])
            logger.info("Deleted long-term fact %s from collection %s", fact_id, category)

    def clear_all(self) -> None:
        """Clear all records from all long-term collections."""
        self._ensure_init()
        for name, col in self._collections.items():
            if col.count() > 0:
                all_data = col.get()
                ids = all_data.get("ids", [])
                if ids:
                    col.delete(ids=ids)
        logger.info("Cleared all long-term collections")

    def close(self) -> None:
        """No explicit close needed for ChromaDB PersistentClient."""
        self._initialized = False
        self._collections.clear()

