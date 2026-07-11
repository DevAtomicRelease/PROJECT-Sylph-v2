"""
Sentence Boundary Splitter
Phase 5.2: Splits streaming LLM tokens into natural sentence clauses

Buffers tokens and emits complete clauses when:
1. A sentence-ending punctuation (. ? !) is followed by a space or end-of-stream
2. OR the buffer exceeds MAX_BUFFER_TOKENS without punctuation (force-flush)
3. Any remaining buffer when the stream ends
"""

import logging
import re
from typing import AsyncGenerator

logger = logging.getLogger("sylph.llm.splitter")

MAX_BUFFER_TOKENS = 60
SENTENCE_ENDERS = re.compile(r"[.!?]$")
# First-clause fast path: a comma/semicolon/colon after enough words is a
# good-enough TTS boundary for the *first* clause of a turn. This cuts
# time-to-first-audio roughly in half on long opening sentences, at zero
# quality cost (Kokoro handles clause fragments naturally).
FIRST_CLAUSE_BREAKERS = re.compile(r"[,;:]$")
FIRST_CLAUSE_MIN_WORDS = 4


class SentenceSplitter:
    """
    Buffers streaming tokens and yields complete sentence clauses
    for routing to the TTS pipeline.
    """

    def __init__(self, max_buffer_tokens: int = MAX_BUFFER_TOKENS):
        self.max_buffer_tokens = max_buffer_tokens

    async def split(
        self, token_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """
        Consume a token stream and yield complete sentence clauses.

        Args:
            token_stream: Async generator yielding individual tokens

        Yields:
            Complete sentence strings, ready for TTS
        """
        buffer = ""
        token_count = 0
        emitted_first = False

        async for token in token_stream:
            buffer += token
            token_count += 1

            # Check for sentence boundary
            stripped = buffer.rstrip()
            is_sentence_end = bool(SENTENCE_ENDERS.search(stripped))
            is_first_clause_break = (
                not emitted_first
                and FIRST_CLAUSE_BREAKERS.search(stripped)
                and len(stripped.split()) >= FIRST_CLAUSE_MIN_WORDS
            )
            if is_sentence_end or is_first_clause_break:
                clause = stripped.strip()
                if clause:
                    logger.debug("Clause emitted (%d tokens): '%s'", token_count, clause[:50])
                    emitted_first = True
                    yield clause
                buffer = ""
                token_count = 0
                continue

            # Force-flush if buffer is too long without punctuation
            if token_count >= self.max_buffer_tokens:
                # Find the last whitespace boundary
                last_space = buffer.rfind(" ")
                if last_space > 0:
                    clause = buffer[:last_space].strip()
                    buffer = buffer[last_space:]
                    token_count = len(buffer.split())
                else:
                    clause = buffer.strip()
                    buffer = ""
                    token_count = 0

                if clause:
                    logger.debug("Force-flushed clause (%d tokens): '%s'", len(clause.split()), clause[:50])
                    yield clause

        # Flush remaining buffer at end of stream
        remainder = buffer.strip()
        if remainder:
            logger.debug("Final flush: '%s'", remainder[:50])
            yield remainder
