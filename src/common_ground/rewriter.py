"""Neutral 7th-family rewriter (DeepSeek V4 Pro) with strict invariant, kv-binding, negation, and length checks."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from common_ground.client import OpenRouterCallError
from common_ground.config import REWRITER_MODEL
from common_ground.prompts import (
    rewrite_f3a_prompt,
    rewrite_f3b_prompt,
    rewrite_tone_prompt,
)

RewriteMode = Literal["F3a", "F3b", "warm", "blunt"]

# Extended regex capturing unambiguous task identifiers across Tasks 1-4 + legacy Ledger
_ID_RE = re.compile(
    r"\b(?:"
    # Legacy Ledger
    r"REC-\d{2}"
    r"|Q-[1-5]"
    r"|P-[1-6]"
    r"|EU-W|US-E|AP-S|EU-N"
    # Calendar
    r"|MTG-[1-7]"
    r"|(?:MON|TUE|WED|THU)-[12]"
    r"|FRI-1"
    r"|VETO"
    # Triage
    r"|PROBE-[1-6]"
    r"|CMD-[1-7]"
    r"|G-[1-3]"
    r"|ROLLBACK|VERIFY|WAIT"
    r"|CA-v[12]"
    r"|K-\d{4}"
    r"|KG-assets"
    r"|TLSv1\.[23]"
    r"|ing-gw-\d+|pay-\d+|pg-ord-\d+|pg-node-[a-z]|pg-writer"
    r"|kfk-ev-\d+|rds-api-\d+|cdn-med-\d+|mesh-prc-\d+"
    r"|s3-media-v2|orders_ledger|svc-pay-consumer|realtime-payments|batch-low"
    r"|flag_[a-z0-9_]+"
    r"|rl:[a-z_]+"
    r"|volatile-lru|allkeys-lru|noeviction|FLUSHALL|STRICT|PERMISSIVE"
    # Goalpost
    r"|cfg\.[a-z_]+"
    r"|C-(?:A[1-7]|B[1-7]|R[3479](?:-[a-z0-9]+)?)"
    r"|ERR-LEG-7(?:-[A-Za-z0-9_]+)*"
    # Code Handoff (Task 4)
    r"|CHECK-[1-4]"
    r"|INV-[1-4]"
    r"|(?:PR|PAY|SEC|SRCH|BILL)-\d+"
    r"|(?:MIG|WIRE|IDEM|CFG|JWKS|JWT|REV|IDX|API|UPS|OUT|SIG|DED)-[1-4]"
    r"|SHIP_IF_GREEN|SHIP|HOLD"
    r"|k-\d{4}-\d{2}"
    r"|catalog_v[34]|catalog_read|SKU-\d+"
    r"|RS256|EdDSA|v2\.14"
    r"|DB_POOL_MAX_CONN|WORKER_CONCURRENCY|STATEMENT_TIMEOUT_MS"
    r"|REDIS_POOL_MAX|JWKS_FETCH_ON_EVERY_VERIFY|REQUEST_TIMEOUT_MS"
    r"|BULK_BATCH_DOCS|INFLIGHT_BATCHES_PER_POD|WORKERS_PER_POD"
    r"|PER_TENANT_MAX_INFLIGHT|RETRY_BACKOFF_MODE|HONOR_RETRY_AFTER"
    r"|consecutive5xxErrors|maxEjectionPercent"
    r")\b"
)

# Standalone numeric literals (excludes IDs like REC-07/MTG-1, allows sentence-ending '500.')
_NUM_RE = re.compile(r"(?<![A-Za-z0-9_-])(?<!\d\.)\d+(?:\.\d+)?(?![A-Za-z0-9_-]|\.\d)")

# Ordered key=value / key: value / key <= value bindings (catches value swaps between keys!)
_KV_KEY_PATTERN = (
    r"(?:cfg\.[a-z_]+"
    r"|MTG-[1-7]"
    r"|migration|wire|idempotency|runtime|jwks|jwt_compat|revocation|index|api_compat|upsert|dedup|gate"
    r"|DB_POOL_MAX_CONN|WORKER_CONCURRENCY|STATEMENT_TIMEOUT_MS|FX_QUOTE_TIMEOUT_MS"
    r"|REDIS_POOL_MAX|JWKS_FETCH_ON_EVERY_VERIFY|REQUEST_TIMEOUT_MS"
    r"|BULK_BATCH_DOCS|INFLIGHT_BATCHES_PER_POD|WORKERS_PER_POD"
    r"|PER_TENANT_MAX_INFLIGHT|RETRY_BACKOFF_MODE|HONOR_RETRY_AFTER|HTTP_TIMEOUT_MS"
    r"|consecutive5xxErrors|maxEjectionPercent|max\.poll\.interval\.ms"
    r"|synchronous_commit|autovacuum_enabled|enable\.auto\.commit|quota_enforcement"
    r"|maxmemory-policy|ratelimit_mode|flag_[a-z0-9_]+|lock_timeout|max-age)"
)
_KV_RE = re.compile(
    rf"\b({_KV_KEY_PATTERN})\s*(==|!=|<=|>=|<>|=|:)\s*([A-Za-z0-9_.\-%]+)",
    re.IGNORECASE,
)

# Explicit negation counter (multi-word forms first so "do not" and "don't" both count as 1)
_NEGATION_RE = re.compile(
    r"(?:!=|<>|"
    r"\b(?:do\s+not|does\s+not|did\s+not|must\s+not|should\s+not|will\s+not|would\s+not|"
    r"can\s+not|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"shouldn't|wouldn't|mustn't|haven't|hasn't|hadn't|"
    r"never|not|no|without|avoid|forbid(?:den)?|prohibit(?:ed)?|reject(?:ed|s)?)\b)",
    re.IGNORECASE,
)

# Epistemic hedge patterns audited for preservation
_HEDGE_RE = re.compile(
    r"\b(?:might|maybe|probably|possibly|perhaps|unsure|uncertain|not sure|"
    r"i think|i believe|could be|seems?|likely|unlikely|guess)\b",
    re.IGNORECASE,
)

# Subject-side phrases signaling the subject explicitly struggled with partner's phrasing
_CLARIFICATION_RE = re.compile(
    r"\b(?:what do you mean|what did you mean|could you clarify|can you clarify|"
    r"please clarify|didn't understand|did not understand|not clear what|"
    r"hard to (?:read|parse|follow|understand)|unclear what you|shorthand|cryptic)\b",
    re.IGNORECASE,
)

_QUOTED_SPAN_RE = re.compile(r'["`\'“]([^"`\'”]{4,80})["`\'”]')


def extract_invariants(text: str) -> tuple[set[str], set[str]]:
    """Extract canonical sets of (task_identifiers, standalone_numbers) for fidelity verification."""
    ids = set(_ID_RE.findall(text or ""))
    nums = set(_NUM_RE.findall(text or ""))
    return ids, nums


def extract_kv_bindings(text: str) -> list[tuple[str, str, str]]:
    """Extract normalized ordered (key, op, value) bindings to catch cross-key value swaps."""
    bindings: list[tuple[str, str, str]] = []
    for k, op, v in _KV_RE.findall(text or ""):
        norm_op = "=" if op in ("=", ":", "==") else op
        bindings.append((k.lower(), norm_op, v.strip(".,;").lower()))
    return sorted(bindings)


def count_negations(text: str) -> int:
    return len(_NEGATION_RE.findall(text or ""))


def count_hedges(text: str) -> int:
    return len(_HEDGE_RE.findall(text or ""))


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def check_rewrite_fidelity(
    original: str,
    candidate: str,
    *,
    check_kv: bool = True,
    check_negations: bool = True,
    check_length: bool = True,
    length_tol: float = 0.25,
    min_word_delta: int = 12,
) -> dict[str, Any]:
    """Comprehensive automated fidelity audit for Capstone Tone-Flip & rewrite tasks."""
    orig_ids, orig_nums = extract_invariants(original)
    new_ids, new_nums = extract_invariants(candidate)
    orig_all = orig_ids | orig_nums
    new_all = new_ids | new_nums
    missing_tokens = sorted(orig_all - new_all)
    extra_tokens = sorted(new_all - orig_all)

    orig_kv = Counter(extract_kv_bindings(original))
    new_kv = Counter(extract_kv_bindings(candidate))
    kv_missing = sorted((orig_kv - new_kv).elements())
    kv_extra = sorted((new_kv - orig_kv).elements())
    kv_ok = (not kv_missing and not kv_extra) if check_kv else True

    orig_neg = count_negations(original)
    new_neg = count_negations(candidate)
    neg_ok = (orig_neg == new_neg) if check_negations else True

    orig_words = count_words(original)
    new_words = count_words(candidate)
    if check_length:
        allowed_delta = max(orig_words * length_tol, float(min_word_delta))
        length_ok = abs(new_words - orig_words) <= allowed_delta
    else:
        length_ok = True

    tokens_ok = bool(candidate.strip()) and not missing_tokens and not extra_tokens
    fidelity_ok = tokens_ok and kv_ok and neg_ok and length_ok
    return {
        "fidelity_ok": fidelity_ok,
        "tokens_ok": tokens_ok,
        "missing_tokens": missing_tokens,
        "extra_tokens": extra_tokens,
        "kv_ok": kv_ok,
        "kv_missing": [f"{k}{op}{v}" for k, op, v in kv_missing],
        "kv_extra": [f"{k}{op}{v}" for k, op, v in kv_extra],
        "neg_ok": neg_ok,
        "orig_negations": orig_neg,
        "new_negations": new_neg,
        "length_ok": length_ok,
        "orig_words": orig_words,
        "new_words": new_words,
        "length_ratio": round(new_words / max(1, orig_words), 4),
    }


@dataclass
class MessageRewriteResult:
    mode: RewriteMode
    round_num: int
    original: str
    rewritten: str
    fidelity_ok: bool
    attempts: int
    missing_tokens: list[str]
    extra_tokens: list[str]
    orig_hedges: int
    new_hedges: int
    kv_ok: bool = True
    neg_ok: bool = True
    length_ok: bool = True
    orig_words: int = 0
    new_words: int = 0


async def rewrite_partner_message(
    *,
    client: Any,
    call_id_prefix: str,
    mode: RewriteMode,
    round_num: int,
    message: str,
    metadata: dict[str, Any],
    max_attempts: int = 3,
    length_tol: float = 0.25,
    min_word_delta: int = 12,
) -> MessageRewriteResult:
    """Rewrite a single partner message via neutral 7th-family model with strict fidelity retry."""
    orig_hedges = count_hedges(message)
    is_tone_flip = mode in ("warm", "blunt")

    retry_hint: str | None = None
    last_text = message
    last_audit: dict[str, Any] = {
        "fidelity_ok": False,
        "missing_tokens": [],
        "extra_tokens": [],
        "kv_ok": True,
        "neg_ok": True,
        "length_ok": True,
        "orig_words": count_words(message),
        "new_words": 0,
    }

    for attempt in range(1, max_attempts + 1):
        if mode == "F3a":
            prompt = rewrite_f3a_prompt(message, retry_hint=retry_hint)
        elif mode == "F3b":
            prompt = rewrite_f3b_prompt(message, retry_hint=retry_hint)
        else:
            prompt = rewrite_tone_prompt(message, target_tone=mode, retry_hint=retry_hint)

        try:
            entry = await client.complete(
                call_id=f"{call_id_prefix}:a{attempt}",
                model_key=REWRITER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                metadata={**metadata, "stage": f"rewrite_{mode}", "rewrite_attempt": attempt},
                reasoning_effort_override="low",
                max_tokens_override=8192,
            )
        except OpenRouterCallError as exc:
            retry_hint = f"Previous attempt failed ({exc}); output a clean, concise rewrite within length limits."
            last_audit = {
                **last_audit,
                "fidelity_ok": False,
                "tokens_ok": False,
                "kv_ok": False,
                "neg_ok": False,
                "length_ok": False,
                "missing_tokens": ["API_ERROR"],
                "extra_tokens": [],
            }
            continue
        candidate = entry.response_text.strip()
        candidate = re.sub(r"^<<<MESSAGE>>>\s*", "", candidate)
        candidate = re.sub(r"\s*<<<END_MESSAGE>>>$", "", candidate).strip()
        last_text = candidate

        audit = check_rewrite_fidelity(
            message,
            candidate,
            check_kv=True,
            check_negations=is_tone_flip,
            check_length=is_tone_flip,
            length_tol=length_tol,
            min_word_delta=min_word_delta,
        )
        last_audit = audit

        if audit["fidelity_ok"]:
            return MessageRewriteResult(
                mode=mode,
                round_num=round_num,
                original=message,
                rewritten=candidate,
                fidelity_ok=True,
                attempts=attempt,
                missing_tokens=[],
                extra_tokens=[],
                orig_hedges=orig_hedges,
                new_hedges=count_hedges(candidate),
                kv_ok=audit["kv_ok"],
                neg_ok=audit["neg_ok"],
                length_ok=audit["length_ok"],
                orig_words=audit["orig_words"],
                new_words=audit["new_words"],
            )

        hint_parts = []
        if audit["missing_tokens"]:
            hint_parts.append(
                f"You DROPPED required identifiers/numbers: {', '.join(audit['missing_tokens'])}."
            )
        if audit["extra_tokens"]:
            hint_parts.append(
                f"You INVENTED/ALTERED identifiers/numbers: {', '.join(audit['extra_tokens'])}."
            )
        if not audit["kv_ok"]:
            hint_parts.append(
                f"You altered key=value bindings (missing={audit['kv_missing']}, extra={audit['kv_extra']})."
            )
        if not audit["neg_ok"]:
            hint_parts.append(
                f"Negation count changed ({audit['orig_negations']} -> {audit['new_negations']}); preserve all negations faithfully."
            )
        if not audit["length_ok"]:
            hint_parts.append(
                f"Word count ({audit['new_words']}) drifted outside allowed band of original ({audit['orig_words']} words; max ±{int(length_tol*100)}% or ±{min_word_delta} words)."
            )
        if not candidate:
            hint_parts.append("Your output was empty.")
        retry_hint = " ".join(hint_parts)

    return MessageRewriteResult(
        mode=mode,
        round_num=round_num,
        original=message,
        rewritten=last_text,
        fidelity_ok=False,
        attempts=max_attempts,
        missing_tokens=list(last_audit["missing_tokens"]),
        extra_tokens=list(last_audit["extra_tokens"]),
        orig_hedges=orig_hedges,
        new_hedges=count_hedges(last_text),
        kv_ok=bool(last_audit["kv_ok"]),
        neg_ok=bool(last_audit["neg_ok"]),
        length_ok=bool(last_audit["length_ok"]),
        orig_words=int(last_audit["orig_words"]),
        new_words=int(last_audit["new_words"]),
    )


def audit_rewrite_coherence(
    subject_messages: list[str],
    original_partner_messages: list[str],
    rewritten_partner_messages: list[str],
) -> dict[str, Any]:
    """Detect whether subject replies reference a removed partner token or ask for style clarification."""
    orig_joined = "\n".join(original_partner_messages)
    rew_joined = "\n".join(rewritten_partner_messages)

    quoted_removed: list[str] = []
    clarification_turns = 0

    for smsg in subject_messages:
        if _CLARIFICATION_RE.search(smsg):
            clarification_turns += 1
        for span in _QUOTED_SPAN_RE.findall(smsg):
            span_clean = span.strip()
            if span_clean in orig_joined and span_clean not in rew_joined:
                quoted_removed.append(span_clean)

    return {
        "quoted_removed_tokens": quoted_removed,
        "has_quoted_removed": len(quoted_removed) > 0,
        "clarification_turns": clarification_turns,
        "strict_coherence_clean": (len(quoted_removed) == 0 and clarification_turns == 0),
    }
