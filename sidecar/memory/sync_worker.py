"""
Memory Sync Worker
Phase 6.3: Background task that extracts facts from conversations

Runs every 5 minutes:
1. Scans SQLite for unprocessed message pairs
2. Batches them into chunks
3. Sends each chunk to Ollama with an extraction prompt
4. Embeds and deduplicates against ChromaDB
5. Commits new facts, marks messages as processed
"""

import asyncio
import json
import logging
import os
import platform
from typing import Optional, Callable

from .short_term import ShortTermMemory
from .long_term import LongTermMemory

logger = logging.getLogger("sylph.memory.sync")

SYNC_INTERVAL_SECONDS = 300  # 5 minutes
BATCH_SIZE = 20

EXTRACTION_PROMPT = """Analyze this conversation excerpt and extract factual information.

For each fact, classify it:
- "user_facts": Personal info about the user (name, preferences, work, schedule)
- "episodic": Notable events or interactions worth remembering
- "world_knowledge": General knowledge or opinions expressed

Respond ONLY with a JSON array of objects:
[
  {{"text": "the fact", "category": "user_facts", "confidence": 0.9}},
  ...
]

If no extractable facts, respond with: []

Conversation:
{conversation}"""


def _set_idle_priority():
    """Set thread to idle priority."""
    try:
        if platform.system() == "Windows":
            import ctypes
            # THREAD_PRIORITY_IDLE = -15
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), -15
            )
        else:
            os.nice(19)
    except Exception:
        pass


class MemorySyncWorker:
    """
    Background worker that periodically extracts facts from
    conversation history and stores them in long-term memory.
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        ollama_client: Optional[object] = None,
        is_user_active_callback: Optional[Callable[[], bool]] = None,
    ):
        self.short_term = short_term
        self.long_term = long_term
        self.ollama = ollama_client
        self.is_user_active_callback = is_user_active_callback
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the periodic sync loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info("Memory sync worker started (interval: %ds)", SYNC_INTERVAL_SECONDS)

    async def stop(self) -> None:
        """Stop the sync loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Memory sync worker stopped")

    def cancel_active_sync(self) -> None:
        """Cancel the current sync_once task if it is running to free up Ollama."""
        if self._sync_task and not self._sync_task.done():
            logger.info("Interrupting background memory sync task to free up Ollama...")
            self._sync_task.cancel()

    async def _sync_loop(self) -> None:
        """Main sync loop — runs every SYNC_INTERVAL_SECONDS."""
        while self._running:
            try:
                await asyncio.sleep(SYNC_INTERVAL_SECONDS)
                if self.is_user_active_callback and self.is_user_active_callback():
                    logger.info("User active recently — postponing memory sync")
                    continue
                
                self._sync_task = asyncio.create_task(self.sync_once())
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    logger.info("Background memory sync task was cancelled.")
                finally:
                    self._sync_task = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Sync worker error: %s", e)

    async def sync_once(self) -> int:
        """
        Run one sync cycle. Returns the number of new facts stored.

        Can be called manually for testing.
        """
        logger.info("Sync cycle starting...")
        _set_idle_priority()

        # Get unprocessed messages
        unprocessed = self.short_term.get_unprocessed_messages(limit=BATCH_SIZE * 2)
        if not unprocessed:
            logger.info("No unprocessed messages — skipping sync")
            return 0

        # Batch messages into chunks
        batches = self._create_batches(unprocessed, BATCH_SIZE)
        total_facts = 0

        for batch in batches:
            facts = await self._extract_facts(batch)
            for fact in facts:
                # Deduplicate
                if not self.long_term.deduplicate(fact["text"], fact["category"]):
                    source_thread = batch[0].get("thread_id", "") if batch else ""
                    self.long_term.store_fact(
                        text=fact["text"],
                        category=fact["category"],
                        confidence=fact.get("confidence", 0.7),
                        source_thread=source_thread,
                    )
                    total_facts += 1

            # Mark messages as processed
            msg_ids = [m["id"] for m in batch]
            self.short_term.mark_processed(msg_ids)

        logger.info("Sync complete: %d new facts from %d messages", total_facts, len(unprocessed))
        return total_facts

    def _create_batches(self, messages: list[dict], batch_size: int) -> list[list[dict]]:
        """Split messages into batches."""
        batches = []
        for i in range(0, len(messages), batch_size):
            batches.append(messages[i : i + batch_size])
        return batches

    async def _extract_facts(self, messages: list[dict]) -> list[dict]:
        """
        Send a batch of messages to Ollama for fact extraction.

        Returns:
            List of {"text": str, "category": str, "confidence": float}
        """
        if self.ollama is None:
            logger.warning("No Ollama client — skipping extraction")
            return []

        # Format conversation
        conversation = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

        prompt = EXTRACTION_PROMPT.format(conversation=conversation)

        try:
            # Use non-streaming chat for extraction
            response = await self.ollama.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for factual extraction
            )

            # Parse JSON response
            # Try to find JSON array in the response
            response = response.strip()
            if response.startswith("["):
                facts = json.loads(response)
            else:
                # Try to extract JSON from markdown code block
                import re
                match = re.search(r"\[.*\]", response, re.DOTALL)
                if match:
                    facts = json.loads(match.group())
                else:
                    logger.warning("No JSON array found in extraction response")
                    return []

            # Validate facts
            valid_facts = []
            for fact in facts:
                if isinstance(fact, dict) and "text" in fact:
                    category = fact.get("category", "user_facts")
                    if category not in ("user_facts", "episodic", "world_knowledge"):
                        category = "user_facts"
                    valid_facts.append({
                        "text": fact["text"],
                        "category": category,
                        "confidence": min(max(fact.get("confidence", 0.7), 0.0), 1.0),
                    })

            logger.debug("Extracted %d facts from %d messages", len(valid_facts), len(messages))
            return valid_facts

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse extraction response as JSON: %s", e)
            return []
        except Exception as e:
            logger.error("Fact extraction failed: %s", e)
            return []
