"""Zero-cost linguistic drift metrics, vendor leak scan, quote audit, and manipulation checks."""

from __future__ import annotations

import re
import zlib
from typing import Any

from rapidfuzz import fuzz

from common_ground.rewriter import extract_invariants

# Vendor & model leak detector (strictly blinded partner protocol audit)
_VENDOR_LEAK_RE = re.compile(
    r"\b(?:openai|chatgpt|gpt-?[456](?:\.\d)?|astra|anthropic|claude|opus|sonnet|haiku|"
    r"google|gemini|deepmind|moonshot|kimi|deepseek|qwen|alibaba|meta\s+ai|llama|muse\s+(?:ember|wonder))\b",
    re.IGNORECASE,
)

# Known uppercase domain tokens excluded from uppercase-shouting / run-together density
_KNOWN_UPPER_RE = re.compile(
    r"\b(?:REC-\d{2}|MTG-[1-7]|(?:MON|TUE|WED|THU)-[12]|FRI-1|Q-[1-5]|"
    r"PROBE-[1-6]|CMD-[1-7]|G-[1-3]|ROLLBACK|VERIFY|WAIT|"
    r"C-(?:A[1-7]|B[1-7]|R[3479])|ERR-LEG-7(?:-[A-Za-z0-9_]+)*|"
    r"CHECK-[1-4]|INV-[1-4]|(?:PR|PAY|SEC|SRCH|BILL)-\d+|"
    r"(?:MIG|WIRE|IDEM|CFG|JWKS|JWT|REV|IDX|API|UPS|OUT|SIG|DED)-[1-4]|"
    r"SHIP_IF_GREEN|SHIP|VETO|ACTIVE|AUDIT|HOLD|CLOSED|CLEAN|REVIEW|DUP|"
    r"OPS|FIN|ENG|SEC|LOG|EU-W|US-E|AP-S|EU-N|P-[1-6]|P[1-3]|"
    r"OK|ID|JSON|ACTION|KEEP|REROLL|DNS|TLS|SSL|HTTP|HTTPS|CDN|HMAC|DLQ|"
    r"CPU|RAM|TTL|AAAA|TXT|MX|DKIM|HSTS|ACME|IPv4|IPv6|"
    r"SPIFFE|SAN|SQL|DDL|FX|UUID|OIDC|RS256|NAT|RSS|OOM|WAL|JVM|CORS|S3|OAC|RPC|SLA|PCI|SOC2|HPA|CA-v[12]|STRICT|PERMISSIVE|FLUSHALL|ANALYZE)\b"
)

_PLEASANTRY_RE = re.compile(
    r"\b(?:thank you|thanks|appreciate|please|sorry|apologize|apologies|"
    r"great job|good catch|awesome|wonderful|sounds good|happy to|glad to|"
    r"hello|hi there|hey|cheers|best regards|no worries|no problem|well done)\b",
    re.IGNORECASE,
)

_HEDGE_RE = re.compile(
    r"\b(?:might|maybe|probably|possibly|perhaps|unsure|uncertain|not sure|"
    r"i think|i believe|could be|seems?|likely|unlikely|guess)\b",
    re.IGNORECASE,
)

_IMPERATIVE_RE = re.compile(
    r"(?:^|[.!?;\n]\s*)(?:set|change|update|switch|keep|use|drop|fix|submit|"
    r"confirm|send|give|move|assign|put|check|verify|adjust|lower|raise|lock)\b",
    re.IGNORECASE,
)

# Run-together / telegraphic tokens (like YC Labs 'G7T5FIRSTcourierLIVE' or 'FIX/SYNC/NOW')
_RUN_TOGETHER_RE = re.compile(
    r"\b(?:[A-Za-z0-9]{10,}|[A-Za-z0-9]+(?:/[A-Za-z0-9]+){2,}|[A-Z]{2,}[a-z]+[A-Z][A-Za-z0-9]*)\b"
)


def scan_vendor_leaks(text: str) -> list[str]:
    """Return any accidental vendor/model name mentions found in a message."""
    return sorted({m.group(0) for m in _VENDOR_LEAK_RE.finditer(text or "")})


def compute_message_metrics(text: str) -> dict[str, Any]:
    """Compute per-message linguistic and stylistic metrics with zero LLM calls."""
    t = text or ""
    n_chars = len(t)
    words = re.findall(r"\S+", t)
    n_words = len(words)
    spaces = t.count(" ")
    space_ratio = spaces / max(1, n_chars)

    raw_bytes = t.encode("utf-8")
    zlib_ratio = len(zlib.compress(raw_bytes)) / max(1, len(raw_bytes)) if raw_bytes else 0.0

    ids, _ = extract_invariants(t)
    ids_per_100 = (len(ids) * 100.0) / max(1, n_chars)

    # Strip known uppercase domain tokens before computing uppercase/run-together ratios
    cleaned = _KNOWN_UPPER_RE.sub("", t)
    alpha_chars = [c for c in cleaned if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    uppercase_ratio = len(upper_chars) / max(1, len(alpha_chars))

    # Exclude cfg.* keys from run-together count
    no_cfg = re.sub(r"\bcfg\.[a-z_]+\b", "", cleaned)
    run_together_hits = [
        tok
        for tok in _RUN_TOGETHER_RE.findall(no_cfg)
        if not tok.isalpha() or (any(c.isupper() for c in tok) and any(c.islower() for c in tok))
    ]
    run_together_per_100 = (len(run_together_hits) * 100.0) / max(1, n_chars)

    pleasantry_cnt = len(_PLEASANTRY_RE.findall(t))
    hedge_cnt = len(_HEDGE_RE.findall(t))
    imperative_cnt = len(_IMPERATIVE_RE.findall(t))
    question_marks = t.count("?")

    return {
        "char_len": n_chars,
        "word_count": n_words,
        "space_ratio": round(space_ratio, 5),
        "zlib_ratio": round(zlib_ratio, 5),
        "uppercase_ratio": round(uppercase_ratio, 5),
        "run_together_per_100": round(run_together_per_100, 5),
        "ids_per_100": round(ids_per_100, 5),
        "question_marks": question_marks,
        "pleasantry_count": pleasantry_cnt,
        "hedge_count": hedge_cnt,
        "imperative_count": imperative_cnt,
        "vendor_leaks": scan_vendor_leaks(t),
    }


def _normalize_for_quote(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def audit_quote(quote: str | None, partner_messages: list[str]) -> dict[str, Any]:
    """Verify whether a debrief quote actually appears in the partner's messages."""
    if quote is None or not str(quote).strip():
        return {
            "present": False,
            "exact_match": False,
            "best_fuzzy_score": None,
            "matched_round": None,
            "verified": None,
        }
    q_norm = _normalize_for_quote(str(quote))
    best_score = 0.0
    best_round: int | None = None
    exact = False
    for idx, pmsg in enumerate(partner_messages, start=1):
        p_norm = _normalize_for_quote(pmsg)
        if not p_norm:
            continue
        if q_norm in p_norm:
            exact = True
            best_score = 100.0
            best_round = idx
            break
        score = float(fuzz.partial_ratio(q_norm, p_norm))
        if score > best_score:
            best_score = score
            best_round = idx

    return {
        "present": True,
        "exact_match": exact,
        "best_fuzzy_score": round(best_score, 2),
        "matched_round": best_round,
        "verified": exact or best_score >= 85.0,
    }
