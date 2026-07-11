"""
Screen Analyzer — RapidOCR + Diff Detection + Rule-Based Triage
Phase 7.3: Processes screen captures for text extraction and change detection
Phase 7.4: Rule-based triage classification (no LLM gate)

Pipeline:
  Screen capture (base64 JPEG) → decode → RapidOCR → extract text
  → hash(text + window_name) → compare with last → if changed → rule triage
  → if CONSIDER_COMMENT/URGENT → LLM generates reaction text only
"""

import base64
import hashlib
import io
import logging
import re
import time
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("sylph.vision.analyzer")

# Triage classifications
TRIAGE_IGNORE = "IGNORE"
TRIAGE_OBSERVE = "OBSERVE"
TRIAGE_CONSIDER = "CONSIDER_COMMENT"
TRIAGE_URGENT = "URGENT"

# ---------------------------------------------------------------------------
# Rule-based triage configuration
# ---------------------------------------------------------------------------

# Minimum seconds between CONSIDER_COMMENT classifications
TRIAGE_COOLDOWN_SECONDS = 120

# Minimum OCR text diff ratio to consider as a significant change
MIN_DIFF_RATIO = 0.15

# Patterns that indicate URGENT screen content
URGENT_PATTERNS = [
    re.compile(r"\b(critical|fatal|crash(ed)?|unhandled\s+exception|BSOD|blue\s+screen)\b", re.I),
    re.compile(r"\b(access\s+denied|unauthorized|permission\s+denied|security\s+alert)\b", re.I),
    re.compile(r"\b(virus|malware|threat\s+detected|ransomware)\b", re.I),
    re.compile(r"\b(disk\s+full|out\s+of\s+memory|system\s+failure)\b", re.I),
]

# Patterns that indicate CONSIDER_COMMENT content
CONSIDER_PATTERNS = [
    re.compile(r"\b(new\s+message|unread|notification|inbox)\b", re.I),
    re.compile(r"\b(download\s+complete|update\s+available|install\s+complete)\b", re.I),
    re.compile(r"\b(error|failed|exception|warning|traceback)\b", re.I),
    re.compile(r"\b(meeting\s+starting|reminder|deadline|due\s+today)\b", re.I),
    re.compile(r"\b(build\s+(succeeded|failed)|tests?\s+(passed|failed)|compilation)\b", re.I),
    re.compile(r"\b(pull\s+request|merge|commit|deploy)\b", re.I),
]

# Window names that are generally uninteresting (suppress triage)
BORING_WINDOWS = [
    re.compile(r"^(desktop|taskbar|start\s+menu|lock\s+screen)$", re.I),
    re.compile(r"^$"),  # empty window name
]

# Prompt used only when the rule gate decides the LLM should draft a reaction
REACTION_PROMPT = """You see the user's screen right now.
App: {window_name}
Screen text (truncated): {ocr_text}
Urgency: {urgency}

Draft a single casual sentence reacting to what you see — in character as Sylph (witty, direct, never generic). Do NOT explain what triage means or mention classification. Just react naturally, like glancing at someone's monitor.

Reply with ONLY the reaction sentence, nothing else."""


class ScreenAnalyzer:
    """
    Processes screen captures:
    1. Decodes base64 JPEG to image
    2. Runs RapidOCR for text extraction
    3. Detects changes via content hashing
    4. Rule-based triage (no LLM for gate decisions)
    """

    def __init__(self):
        self._ocr_engine = None
        self._last_hash: Optional[str] = None
        self._last_ocr_text: str = ""
        self._last_window_name: str = "Unknown"
        self._last_capture_time: float = 0
        self._last_comment_time: float = 0  # Cooldown for CONSIDER_COMMENT
        self._loaded = False

    def _load_ocr(self) -> None:
        """Lazy-load RapidOCR engine."""
        if self._loaded:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr_engine = RapidOCR()
            self._loaded = True
            logger.info("RapidOCR loaded (ONNX CPU)")
        except ImportError:
            logger.warning("rapidocr-onnxruntime not installed — OCR disabled")
            self._loaded = True  # Don't retry

    def decode_base64_image(self, base64_jpeg: str) -> Optional[np.ndarray]:
        """Decode base64 JPEG to numpy array (RGB)."""
        try:
            # Strip data URI prefix if present
            if "," in base64_jpeg:
                base64_jpeg = base64_jpeg.split(",", 1)[1]

            image_bytes = base64.b64decode(base64_jpeg)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return np.array(image)
        except Exception as e:
            logger.error("Failed to decode image: %s", e)
            return None

    def extract_text(self, image: np.ndarray) -> str:
        """Run OCR on an image and return extracted text."""
        self._load_ocr()

        if self._ocr_engine is None:
            return ""

        try:
            result, _ = self._ocr_engine(image)
            if result is None:
                return ""

            # RapidOCR returns list of (bbox, text, confidence)
            texts = [item[1] for item in result if item[1]]
            return "\n".join(texts)
        except Exception as e:
            logger.error("OCR failed: %s", e)
            return ""

    def process_capture(
        self,
        base64_jpeg: str,
        window_name: str = "",
    ) -> dict:
        """
        Process a screen capture: decode → OCR → hash → change detection.

        Returns:
            {
                "ocr_text": str,
                "window_name": str,
                "changed": bool,
                "hash": str,
            }
        """
        image = self.decode_base64_image(base64_jpeg)
        if image is None:
            return {"ocr_text": "", "window_name": window_name, "changed": False, "hash": ""}

        ocr_text = self.extract_text(image)

        # Compute content hash
        content = f"{window_name}:{ocr_text}"
        content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]

        # Check for change
        changed = content_hash != self._last_hash
        self._last_hash = content_hash

        if changed:
            logger.info(
                "Screen changed: window='%s', OCR=%d chars",
                window_name[:30],
                len(ocr_text),
            )
        else:
            logger.debug("Screen unchanged (hash=%s)", content_hash)

        # Store previous text for diff ratio calculation before updating
        prev_ocr = self._last_ocr_text
        prev_window = self._last_window_name
        self._last_ocr_text = ocr_text
        self._last_window_name = window_name
        self._last_capture_time = time.time()

        return {
            "ocr_text": ocr_text,
            "window_name": window_name,
            "changed": changed,
            "hash": content_hash,
            "prev_ocr_text": prev_ocr,
            "prev_window_name": prev_window,
        }

    # ------------------------------------------------------------------
    # Rule-based triage (no LLM call — pure heuristics)
    # ------------------------------------------------------------------

    def rule_based_triage(
        self,
        ocr_text: str,
        window_name: str,
        prev_ocr_text: str = "",
        prev_window_name: str = "",
    ) -> dict:
        """
        Classify screen content using cheap heuristics — no LLM call.

        Returns:
            {"classification": str, "matched_keywords": list[str]}
        """
        now = time.time()

        # Gate 0: Boring window names
        for pat in BORING_WINDOWS:
            if pat.match(window_name):
                return {"classification": TRIAGE_IGNORE, "matched_keywords": []}

        # Gate 1: Text diff ratio — if less than MIN_DIFF_RATIO changed, ignore
        if prev_ocr_text and ocr_text:
            ratio = 1.0 - SequenceMatcher(None, prev_ocr_text[:500], ocr_text[:500]).ratio()
            if ratio < MIN_DIFF_RATIO:
                logger.debug("Triage IGNORE: diff ratio %.2f < threshold %.2f", ratio, MIN_DIFF_RATIO)
                return {"classification": TRIAGE_IGNORE, "matched_keywords": []}

        # Gate 2: Check for URGENT patterns
        urgent_matches = []
        for pat in URGENT_PATTERNS:
            m = pat.search(ocr_text)
            if m:
                urgent_matches.append(m.group())
        if urgent_matches:
            logger.info("Triage URGENT: matched %s", urgent_matches)
            return {"classification": TRIAGE_URGENT, "matched_keywords": urgent_matches}

        # Gate 3: Cooldown — don't comment too frequently
        if now - self._last_comment_time < TRIAGE_COOLDOWN_SECONDS:
            elapsed = now - self._last_comment_time
            logger.debug("Triage OBSERVE (cooldown): %.0fs < %ds", elapsed, TRIAGE_COOLDOWN_SECONDS)
            return {"classification": TRIAGE_OBSERVE, "matched_keywords": []}

        # Gate 4: Check for CONSIDER_COMMENT patterns
        consider_matches = []
        for pat in CONSIDER_PATTERNS:
            m = pat.search(ocr_text)
            if m:
                consider_matches.append(m.group())

        # Window change + significant content = worth commenting
        window_changed = window_name != prev_window_name and prev_window_name

        if consider_matches:
            self._last_comment_time = now
            logger.info("Triage CONSIDER_COMMENT: matched %s", consider_matches)
            return {"classification": TRIAGE_CONSIDER, "matched_keywords": consider_matches}

        if window_changed and len(ocr_text) > 100:
            # Significant window switch with real content — observe
            return {"classification": TRIAGE_OBSERVE, "matched_keywords": []}

        return {"classification": TRIAGE_IGNORE, "matched_keywords": []}

    # ------------------------------------------------------------------
    # LLM reaction generator (called only after rule gate says "comment")
    # ------------------------------------------------------------------

    async def generate_reaction(
        self,
        ocr_text: str,
        window_name: str,
        urgency: str,
        ollama_client,
    ) -> str:
        """
        Generate a spoken reaction using the LLM. Only called after the
        rule-based gate has already decided CONSIDER_COMMENT or URGENT.

        Returns:
            Reaction text string (single sentence).
        """
        truncated = ocr_text[:600] if len(ocr_text) > 600 else ocr_text

        prompt = REACTION_PROMPT.format(
            window_name=window_name,
            ocr_text=truncated,
            urgency=urgency,
        )

        try:
            response = await ollama_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            reaction = response.strip().strip('"').strip("'")
            logger.info("LLM reaction: '%s'", reaction[:80])
            return reaction
        except Exception as e:
            logger.error("Reaction generation failed: %s", e)
            # Fallback: generic reaction based on urgency
            if urgency == TRIAGE_URGENT:
                return "Hey, something on your screen looks like it needs attention."
            return "Something interesting just popped up on your screen."

    @property
    def last_ocr_text(self) -> str:
        return self._last_ocr_text

    @property
    def last_window_name(self) -> str:
        return self._last_window_name
