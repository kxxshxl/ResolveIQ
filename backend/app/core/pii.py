"""Regex-based PII redaction applied before storage, embedding, logging and LLM prompts.

This is a deliberately conservative, dependency-free baseline (emails, phones, card numbers,
IPs, labelled account/customer ids, national-id style numbers). It does not detect person
names or street addresses - production should add an NER-based scrubber (see docs/production.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_ACCOUNT = re.compile(
    r"\b(?:account|acct|a/c|customer|cust|subscriber|ref(?:erence)?|order)\s*(?:no\.?|number|id|#)?\s*[:#-]?\s*[A-Z]{0,4}[- ]?\d{5,}\b",
    re.I,
)
_NATIONAL_ID = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{4}\s\d{4}\s\d{4}\b")
_PHONE = re.compile(r"(?<![\w.])(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}(?!\w|\.\d)")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    counts: dict[str, int] = {}

    def sub(pattern: re.Pattern, label: str, s: str, check=None) -> str:
        def repl(m: re.Match) -> str:
            if check and not check(m.group(0)):
                return m.group(0)
            counts[label] = counts.get(label, 0) + 1
            return f"[{label}]"

        return pattern.sub(repl, s)

    s = sub(_EMAIL, "EMAIL", text)
    s = sub(_CARD, "CARD", s, check=lambda x: _luhn_ok(re.sub(r"\D", "", x)))
    s = sub(_ACCOUNT, "ACCOUNT_ID", s)
    s = sub(_NATIONAL_ID, "NATIONAL_ID", s)
    s = sub(_IPV4, "IP_ADDRESS", s)
    # phones: require >= 9 digits so times/amounts ("8 PM", "45.00", "2 times") are untouched
    s = sub(_PHONE, "PHONE", s, check=lambda x: len(re.sub(r"\D", "", x)) >= 9)
    return RedactionResult(text=s, counts=counts)
