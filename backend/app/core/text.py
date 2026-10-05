from __future__ import annotations

import html
import re
import unicodedata

_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# Phrases that try to steer the LLM instead of describing a telecom problem.
_INJECTION = re.compile(
    r"(ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)|disregard (the )?(system|previous)|"
    r"you are now|reveal (your|the) (system )?prompt|jailbreak|do anything now)",
    re.I,
)


def normalize_text(text: str) -> str:
    """Unicode-normalise, strip markup/control chars, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(_TAGS.sub(" ", text))
    text = _CTRL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))
