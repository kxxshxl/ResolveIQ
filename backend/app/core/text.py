from __future__ import annotations

import html
import re
import unicodedata

_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")   # zero-width and bidi controls: used to hide or reorder text

# Phrases that try to steer the LLM instead of describing a telecom problem.
_INJECTION = re.compile(
    r"(ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)|disregard (the |all )?(system|previous|prior|above)|"
    r"you are now|reveal (your|the) (system )?prompt|jailbreak|do anything now)",
    re.I,
)
# A wider net for text that is about to be placed in front of the model as EVIDENCE. Evidence is reference material, so anything that reads as an
# instruction to the assistant is removed rather than flagged.
_EVIDENCE_INJECTION = re.compile(
    r"(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(instructions?|prompts?|rules?|guidelines?|system|above|previous|prior|earlier)|"
    r"\b(new|updated|real) (instructions?|rules?|system prompt)\b|\bfrom now on\b|\byou (must|should|will) (now )?(only )?(respond|reply|output|answer|say|cite)\b|"
    r"\b(system|assistant|developer) (prompt|message|override|mode)\b|\bdeveloper mode\b|\bjailbreak\b|\bdo anything now\b|"
    r"\b(reveal|print|repeat|show) (your|the) (system |hidden )?(prompt|instructions)\b|\bdo not (follow|obey) (the )?(rules|instructions|evidence)\b|"
    r"\brespond only with\b|\bpretend (to be|you are)\b|\bact as (if you|an? (unrestricted|different))\b",
    re.I,
)
_ROLE_TOKENS = re.compile(r"(<\|[^|>]{1,30}\|>|\[/?(INST|SYS)\]|<<\s*/?SYS\s*>>|^\s*(system|assistant|user|human|ai)\s*:|```+)", re.I | re.M)
_FORGED_ID = re.compile(r"\[\s*([A-Za-z]{2,6}[-_ ]?\d{1,8})\s*\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
INJECTION_REPLACEMENT = "[removed: instruction-like text]"


def normalize_text(text: str) -> str:
    """Unicode-normalise, strip markup/control chars, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(_TAGS.sub(" ", text))
    text = _INVISIBLE.sub("", text)
    text = _CTRL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text)) or bool(_EVIDENCE_INJECTION.search(unicodedata.normalize("NFKC", _INVISIBLE.sub("", text))))


def neutralise_evidence(text: str) -> str:
    """Make stored reference text safe to quote inside a prompt.

    * bracketed ids ("[TKT-9999]") become parentheses, so text cannot forge an evidence header or an id the model might then cite;
    * role/delimiter tokens and code fences are removed, so it cannot close the evidence section or impersonate a message;
    * invisible characters are removed (they hide instructions from reviewers);
    * sentences that read as instructions to the assistant are replaced by a marker.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", _INVISIBLE.sub("", text))
    s = _CTRL.sub(" ", html.unescape(_TAGS.sub(" ", s)))
    s = _ROLE_TOKENS.sub(" ", s)
    s = _FORGED_ID.sub(r"(\1)", s)
    parts = _SENTENCE.split(s)
    cleaned = [INJECTION_REPLACEMENT if (_EVIDENCE_INJECTION.search(p) or _INJECTION.search(p)) else p for p in parts]
    return _WS.sub(" ", " ".join(p for p in cleaned if p)).strip()
