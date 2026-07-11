"""
Deterministic Intent Router
Runs BEFORE the LLM on every user utterance.

Purpose: commands like "move to the left", "go sit in the corner", "hide",
"come back" must never depend on a 9B model's tool-calling mood. They are
parsed here with strict patterns, executed in <1ms, and answered with a short
in-character acknowledgement — the LLM is bypassed entirely, so the avatar
starts moving while a normal pipeline would still be waiting on first token.

Anything that doesn't match falls through to the planner unchanged, so
ambiguous phrasing ("could you maybe scoot over a bit?") still reaches the
LLM, which can call the `move_avatar` tool instead.
"""

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("sylph.llm.intent_router")


@dataclass
class Intent:
    name: str                      # e.g. "avatar_move"
    params: dict = field(default_factory=dict)
    ack: str = ""                  # short spoken acknowledgement


# Anchors the command verbs so "tell me how to move a file" never matches.
_MOVE_VERB = r"(?:move|go|slide|scoot|shift|walk|float|jump)"
_SELF = r"(?:\s+(?:over|yourself|a\s+bit|a\s+little))?"

_PATTERNS: list[tuple[re.Pattern, str, dict]] = [
    (re.compile(rf"^\s*{_MOVE_VERB}{_SELF}\s+(?:to\s+the\s+)?(?:far\s+)?left(?:\s+(?:side|corner|edge))?\W*$", re.I),
     "avatar_move", {"position": "left"}),
    (re.compile(rf"^\s*{_MOVE_VERB}{_SELF}\s+(?:to\s+the\s+)?(?:far\s+)?right(?:\s+(?:side|corner|edge))?\W*$", re.I),
     "avatar_move", {"position": "right"}),
    (re.compile(rf"^\s*{_MOVE_VERB}{_SELF}\s+(?:to\s+the\s+)?(?:center|middle)(?:\s+of\s+the\s+screen)?\W*$", re.I),
     "avatar_move", {"position": "center"}),
    (re.compile(rf"^\s*(?:{_MOVE_VERB}\s+to\s+the\s+|sit\s+in\s+the\s+)?(?:bottom|lower)[\s-]?left(?:\s+corner)?\W*$", re.I),
     "avatar_move", {"position": "bottom_left"}),
    (re.compile(rf"^\s*(?:{_MOVE_VERB}\s+to\s+the\s+|sit\s+in\s+the\s+)?(?:bottom|lower)[\s-]?right(?:\s+corner)?\W*$", re.I),
     "avatar_move", {"position": "bottom_right"}),
    (re.compile(r"^\s*(?:hide|go\s+away|disappear|get\s+out\s+of\s+(?:the\s+way|my\s+screen))\W*$", re.I),
     "avatar_hide", {}),
    (re.compile(r"^\s*(?:come\s+back|show\s+yourself|reappear|unhide)\W*$", re.I),
     "avatar_show", {}),
    (re.compile(r"^\s*(?:be\s+quiet|quiet\s+mode(?:\s+on)?|shush|stop\s+talking)\W*$", re.I),
     "quiet_on", {}),
    (re.compile(r"^\s*(?:you\s+can\s+talk(?:\s+now)?|quiet\s+mode\s+off|speak\s+freely)\W*$", re.I),
     "quiet_off", {}),
]

_ACKS = {
    "avatar_move": ["On the move.", "Sure, relocating.", "Coming right over.", "Fine, fine — moving."],
    "avatar_hide": ["Poof. Gone.", "Vanishing act, coming up.", "Fine, I'll make myself scarce."],
    "avatar_show": ["Miss me already?", "Back in action.", "I'm here."],
    "quiet_on":    ["Zipping it.", "Going silent."],
    "quiet_off":   ["Finally. I had thoughts.", "Back online, vocally speaking."],
}


def route(user_text: str) -> Optional[Intent]:
    """Return a matched Intent or None if the utterance should go to the LLM."""
    text = user_text.strip()
    if not text or len(text) > 80:
        return None
    for pattern, name, params in _PATTERNS:
        if pattern.match(text):
            ack = random.choice(_ACKS.get(name, [""]))
            logger.info("Deterministic intent matched: %s %s", name, params)
            return Intent(name=name, params=params, ack=ack)
    return None
