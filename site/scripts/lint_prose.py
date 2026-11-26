#!/usr/bin/env python3
"""Deterministic HTAIL editorial & house-style prose linter.

Supports Astro/Markdown (.md), XeLaTeX (.tex), SVG figures (--svg-dir),
and pairwise Markdown parity checks (--compare).

Quote- and code-block-aware: skips verbatim speaker quote bodies (:::quote,
Markdown blockquotes, LaTeX modelquote), fenced code / verbatim listings,
URLs, BibTeX citation brackets, and inline code spans while auditing author
prose, titles, subtitles, abstracts, headings, captions, footnotes, and alt text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Rules & Lexicons
# ---------------------------------------------------------------------------

US_TO_UK: dict[str, str] = {
    "behavior": "behaviour",
    "behaviors": "behaviours",
    "behavioral": "behavioural",
    "favor": "favour",
    "favors": "favours",
    "favored": "favoured",
    "favoring": "favouring",
    "favorable": "favourable",
    "gray": "grey",
    "artifact": "artefact",
    "artifacts": "artefacts",
    "hypothesize": "hypothesise",
    "hypothesizes": "hypothesises",
    "hypothesized": "hypothesised",
    "hypothesizing": "hypothesising",
    "anonymize": "anonymise",
    "anonymizes": "anonymises",
    "anonymized": "anonymised",
    "anonymizing": "anonymising",
    "penalize": "penalise",
    "penalizes": "penalises",
    "penalized": "penalised",
    "penalizing": "penalising",
    "recognize": "recognise",
    "recognizes": "recognises",
    "recognized": "recognised",
    "recognizing": "recognising",
    "summarize": "summarise",
    "summarizes": "summarises",
    "summarized": "summarised",
    "summarizing": "summarising",
    "modeling": "modelling",
    "modeled": "modelled",
    "signaling": "signalling",
    "signaled": "signalled",
    "normalize": "normalise",
    "normalizes": "normalises",
    "normalized": "normalised",
    "normalizing": "normalising",
    "normalization": "normalisation",
    "standardize": "standardise",
    "standardizes": "standardises",
    "standardized": "standardised",
    "standardizing": "standardising",
    "standardization": "standardisation",
    "optimize": "optimise",
    "optimizes": "optimises",
    "optimized": "optimised",
    "optimizing": "optimising",
    "optimization": "optimisation",
    "center": "centre",
    "centered": "centred",
    "centering": "centring",
    "defense": "defence",
    "paralyzed": "paralysed",
    "paralyze": "paralyse",
    "analyze": "analyse",
    "analyzes": "analyses",
    "analyzed": "analysed",
    "analyzing": "analysing",
    "labeled": "labelled",
    "labeling": "labelling",
    "totaled": "totalled",
    "totaling": "totalling",
    "traveled": "travelled",
    "color": "colour",
    "colors": "colours",
    "colored": "coloured",
    "coloring": "colouring",
    "toward": "towards",
    "afterward": "afterwards",
    "unparseable": "unparseable",  # valid UK; kept out via check below
}
US_TO_UK.pop("unparseable", None)

US_SPELLING_RE = re.compile(
    r"\b(" + "|".join(sorted(US_TO_UK.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

LICENSE_NOUN_RE = re.compile(
    r"\b(a|the|permissive|open|text|code|public|same|this|that)\s+(license)\b",
    re.IGNORECASE,
)

PROGRAM_GRANTS_RE = re.compile(
    r"\b(grants?|funding|fellowship|research)\s+(program)\b",
    re.IGNORECASE,
)

BANNED_SLOP_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bdelv(?:e|es|ed|ing)\b", re.I), "ERROR", "Avoid AI-slop verb 'delve'."),
    (re.compile(r"\bunderscor(?:e|es|ed|ing)\b", re.I), "WARN", "Avoid AI-slop booster verb 'underscore'; state the link directly."),
    (re.compile(r"\bspeaks volumes\b", re.I), "ERROR", "Avoid cliché 'speaks volumes'."),
    (re.compile(r"\bserves as a\b", re.I), "WARN", "Avoid filler 'serves as a'; use 'is a' or an active verb."),
    (re.compile(r"\b(?:rich )?tapestry\b", re.I), "ERROR", "Avoid AI-slop cliché 'tapestry'."),
    (re.compile(r"\btestament to\b", re.I), "ERROR", "Avoid AI-slop cliché 'testament to'."),
    (re.compile(r"\bpivotal\b", re.I), "WARN", "Avoid booster adjective 'pivotal'."),
    (re.compile(r"\bsmoking gun\b", re.I), "ERROR", "Avoid sensational cliché 'smoking gun'."),
    (re.compile(r"\bplot twist\b", re.I), "ERROR", "Avoid blog cliché 'plot twist'."),
    (re.compile(r"\bbulletproof\b", re.I), "ERROR", "Avoid hype adjective 'bulletproof'."),
    (re.compile(r"\bstunning\b", re.I), "ERROR", "Avoid effusive booster 'stunning'."),
    (re.compile(r"\bblowouts?\b", re.I), "ERROR", "Avoid informal 'blowout(s)'; use exact operational terms (e.g. 'token-cap failures')."),
    (re.compile(r"\b(?:straight up)\b", re.I), "WARN", "Avoid colloquial register jolt 'straight up'."),
    (re.compile(r"\baforementioned\b", re.I), "WARN", "Avoid bureaucratic filler 'aforementioned'."),
    (re.compile(r"(?:^|[.!?]\s+)(Interestingly|Importantly|Notably|Indeed|More subtly|Perhaps expectedly),?\b"), "WARN", "Avoid throat-clearing adverb opener; let the finding carry itself."),
    (re.compile(r"\bslopmanifesto\.com\b", re.I), "ERROR", "Removed meme reference 'slopmanifesto.com' should not appear."),
]

INTERNAL_LEAK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bannoying[_\s-]models\b", re.I), "Working codename 'annoying_models' leaked."),
    (re.compile(r"Do AI Models Annoy Each Other", re.I), "Retired working title leaked."),
    (re.compile(r"\bmain_7p\b"), "Internal arm identifier 'main_7p' leaked."),
    (re.compile(r"\bopus_boost\b"), "Internal arm identifier 'opus_boost' leaked."),
    (re.compile(r"\bEmpirical Evaluation Report\b"), "Retired working subtitle 'Empirical Evaluation Report' leaked."),
    (re.compile(r"\bv2\.1\b"), "Internal version tag 'v2.1' leaked in prose."),
    (re.compile(r"/Users/[A-Za-z0-9._-]+/"), "Local workstation path '/Users/...' leaked."),
]

DOMAIN_METAPHOR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwithhold(?:s|ing)?\b|\bwithheld\b", re.I), "Use consistent Gricean terminology ('under-informing' / 'under-informed') or 'omitted private rules' for Control 2."),
    (re.compile(r"\bhir(?:e|es|ed|ing)\b", re.I), "Avoid 'hiring' metaphor; use 'teammate preference' / 'preferred'."),
    (re.compile(r"\bstaffing\b", re.I), "Avoid 'staffing' metaphor in author prose (allowed only in verbatim prompt disclosures)."),
    (re.compile(r"\bsurvival\b", re.I), "Avoid 'survival' metaphor; use 'task completion' / 'completion rate'."),
    (re.compile(r"\bgatekeeper\b", re.I), "Avoid 'gatekeeper' metaphor; use 'Release Approver'."),
    (re.compile(r"\bcompute tax\b", re.I), "Avoid 'compute tax' metaphor; state literal cost/token increase."),
    (re.compile(r"\bis infected by\b", re.I), "Avoid scare-quoted 'is infected by'; state behavioural mirroring directly."),
    (re.compile(r"\bmodulo\b", re.I), "Avoid mathematical slang 'modulo' in general prose."),
    (re.compile(r"\bcheap worker(?:s)?\b", re.I), "Prefer 'low-cost worker(s)' over 'cheap worker(s)' for consistency."),
    (re.compile(r"\b(?:e\.g\.|i\.e\.)"), "Consider plain English ('for example', 'that is') instead of Latin abbreviations 'e.g.'/'i.e.' for consistent register."),
]

NICKNAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<!Claude )\bOpus 5\.5\b"), "Use full canonical model name 'Claude Opus 5.5'."),
    (re.compile(r"(?<!Gemini 4 )\bNeon 4\b"), "Use full canonical model name 'Gemini 4 Neon'."),
    (re.compile(r"(?<!Claude )\bHaiku 5\.5\b"), "Use full canonical model name 'Claude Haiku 5.5'."),
    (re.compile(r"\bGPT-6\.2\b(?!\s+Astra)"), "Use full canonical model name 'GPT-6.2 Astra'."),
    (re.compile(r"\bGPT-6\b(?!\.\d|\s+(?:Astra|Luna))"), "Use full canonical model name ('GPT-6 Astra' or 'GPT-6 Luna')."),
    (re.compile(r"(?<!Meta )\bMuse Wonder\b"), "Use full canonical model name 'Meta Muse Wonder'."),
]

OXFORD_COMMA_RE = re.compile(
    r"\b[A-Za-z0-9'-]+(?:\s+[A-Za-z0-9'-]+){0,3},\s+[A-Za-z0-9'-]+(?:\s+[A-Za-z0-9'-]+){0,3},\s+(?:and|or)\s+[A-Za-z0-9'-]+"
)

CONTRASTIVE_HINT_RE = re.compile(
    r"\bnot\s+(?:just\s+|only\s+|merely\s+)?[^.,;:]{2,55},\s+but\b",
    re.IGNORECASE,
)

ZERO_ABSENCE_RE = re.compile(
    r"\bzero\s+(?!(?:or|to|and|vs\.?|of|in|on|at|by|for|from|with|as|is|are|was|were|if|when|than|per|point|sum|shot|based|indexed|mean|variance)\b)[A-Za-z][A-Za-z0-9'-]*",
    re.IGNORECASE,
)

HEADING_PROPER_NOUNS: set[str] = {
    "A", "B", "C", "D", "E", "F", "I",
    "AI", "AA", "API", "LLM", "LLMs", "RL", "CI", "JSON", "PDF", "SVG", "HTML", "CSS",
    "LaTeX", "XeLaTeX", "BibTeX", "MathJax", "KaTeX", "TeX", "Gyre", "Pagella", "Math", "Live",
    "Spectral", "IBM", "Plex", "Mono",
    "HTAIL", "HPI", "ORCID", "arXiv", "GitHub", "HuggingFace", "OpenRouter", "OpenAI", "Google",
    "DeepMind", "Anthropic", "Meta", "Alibaba", "Moonshot", "DeepSeek", "Bespoke", "Labs",
    "GPT", "GPT-6", "GPT-6.2", "GPT-5.2", "Astra", "Luna", "Claude", "Opus", "Haiku",
    "Gemini", "Neon", "Hydrogen", "Kimi", "K3.1", "Pro", "Muse", "Wonder", "Qwen4", "Max", "Flash",
    "Config", "Repair", "Incident", "Triage", "Code", "Handoff", "Calendar", "Negotiation",
    "Pareto", "Grice", "Grice's", "Sutton's", "Clark", "Brennan", "HiddenBench", "CooperBench",
    "MAC", "MultiAgentBench", "Foresight", "Institute", "Grants",
    "Appendix", "Appendices", "Section", "Sections", "Figure", "Figures", "Table", "Tables",
    "Control", "Round", "Rounds", "Player", "Index", "Warm", "Blunt", "Tone", "Rewrite",
    "Missing-State", "Injection", "Bus-Factor", "Decision-Model", "Choice-Logit",
    "Dropout", "Task", "Teammate",
}

SMALL_TITLE_WORDS: set[str] = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "nor", "of",
    "on", "or", "per", "the", "to", "vs", "vs.", "with", "over", "under", "via",
}

PLAIN_MATH_RE = re.compile(
    r"^\s*[+\-−]?\d[\d,{}.]*(?:\s*/\s*\d[\d,{}.]*|\s*\\%|\s*%|\s*\\times|\s*×|\s*(?:pp|w|ms))?\s*$"
)
PLAIN_CI_MATH_RE = re.compile(
    r"^\s*\[\s*[+\-−]?\d[\d.,]*\s*,\s*[+\-−]?\d[\d.,]*\s*\]\s*$"
)


@dataclass
class Issue:
    file: str
    line: int
    severity: str  # ERROR | WARN
    rule: str
    message: str
    snippet: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snippet(text: str, pos: int | None = None, width: int = 90) -> str:
    clean = re.sub(r"\s+", " ", text.strip())
    if len(clean) <= width:
        return clean
    if pos is None:
        return clean[: width - 1] + "…"
    start = max(0, pos - width // 3)
    end = min(len(clean), start + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(clean) else ""
    return prefix + clean[start:end] + suffix


def _mask_inline_quotes(text: str) -> str:
    """Replace short inline double-quoted speech with spaces so verbatim quotes don't false-positive."""
    text = re.sub(r"``(?:(?!'').){1,300}''", lambda m: " " * len(m.group(0)), text)
    text = re.sub(r"“[^”]{1,300}”", lambda m: " " * len(m.group(0)), text)
    text = re.sub(r'"[^"\n]{1,300}"', lambda m: " " * len(m.group(0)), text)
    return text


def _check_heading(file: str, lineno: int, raw_heading: str, issues: list[Issue]) -> None:
    h = raw_heading.strip()
    if not h:
        return
    # Strip trailing inline LaTeX / footnote noise
    h_clean = re.sub(r"\\code\{([^}]*)\}", r"\1", h)
    h_clean = re.sub(r"\\texttt\{([^}]*)\}", r"\1", h_clean)
    h_clean = re.sub(r"\\ref\{[^}]*\}", "1", h_clean)
    h_clean = re.sub(r"~", " ", h_clean)
    h_clean = re.sub(r"`([^`]*)`", r"\1", h_clean).strip()

    if h_clean.endswith(".") and not h_clean.endswith("vs."):
        issues.append(Issue(file, lineno, "ERROR", "heading-trailing-period", "Headings must not end with a full stop.", _snippet(h_clean)))

    # Strip leading section number ("1. ") or "Appendix A: " or "Control 1: "
    body = re.sub(r"^\d+\.\s+", "", h_clean)
    body = re.sub(r"^Appendix\s+[A-Z]\s*:\s*", "", body)
    body = re.sub(r"^Control\s+\d+\s*:\s*", "", body)
    # Remove parentheticals containing paths/file refs
    body_words = re.sub(r"\([^)]*\)", "", body).split()
    if len(body_words) < 3:
        return

    unexpected_caps: list[str] = []
    after_colon = False
    for idx, tok in enumerate(body_words):
        raw_tok = tok.strip(".,:;!?()[]'\"“”`")
        if not raw_tok:
            continue
        if idx == 0 or after_colon:
            after_colon = tok.endswith(":")
            continue
        after_colon = tok.endswith(":")
        # Allow hyphenated proper compounds or known proper nouns / acronyms
        parts = raw_tok.split("-")
        if raw_tok in HEADING_PROPER_NOUNS or all(p in HEADING_PROPER_NOUNS or p.isupper() or p.islower() or p[0].isdigit() for p in parts if p):
            continue
        if raw_tok[0].isupper() and not raw_tok.isupper() and raw_tok.lower() not in SMALL_TITLE_WORDS:
            unexpected_caps.append(raw_tok)

    if len(unexpected_caps) >= 3:
        issues.append(
            Issue(
                file,
                lineno,
                "ERROR",
                "heading-title-case",
                f"Heading appears to use Title Case instead of HTAIL sentence case (capitalised: {', '.join(unexpected_caps[:5])}).",
                _snippet(h_clean),
            )
        )


def _check_math_spans(file: str, lineno: int, line: str, issues: list[Issue]) -> str:
    """Audit inline $...$ spans for plain prose numerals, then return line with math masked."""
    # Ignore escaped \$
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "\\" and i + 1 < n and line[i + 1] == "$":
            out.append("  ")
            i += 2
            continue
        if line[i] == "$":
            if i + 1 < n and line[i + 1] == "$":
                # Display math delimiter on same line
                end = line.find("$$", i + 2)
                if end != -1:
                    out.append(" " * (end + 2 - i))
                    i = end + 2
                    continue
                out.append("  ")
                i += 2
                continue
            # Find closing unescaped single $
            j = i + 1
            while j < n:
                if line[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if line[j] == "$":
                    break
                j += 1
            if j < n and line[j] == "$":
                inner = line[i + 1 : j]
                if inner.strip() and (PLAIN_MATH_RE.match(inner) or PLAIN_CI_MATH_RE.match(inner)):
                    issues.append(
                        Issue(
                            file,
                            lineno,
                            "ERROR",
                            "maths-vs-prose",
                            f"Plain numeral/ratio/percentage/CI inside math mode (${inner}$); set in native prose numerals with Unicode −/× instead.",
                            _snippet(line, i),
                        )
                    )
                out.append(" " * (j + 1 - i))
                i = j + 1
                continue
        out.append(line[i])
        i += 1
    return "".join(out)


def _check_prose_text(
    file: str,
    lineno: int,
    raw_line: str,
    masked_line: str,
    issues: list[Issue],
    *,
    is_latex: bool = False,
    is_note_or_caption: bool = False,
    is_table_row: bool = False,
) -> None:
    # 1. Middle dots (anywhere in line, including masked metadata)
    for pat, label in (
        (r"·", "Unicode middle dot (·)"),
        (r"&middot;|&#183;", "HTML middle dot entity"),
        (r"\\textperiodcentered\b", r"LaTeX \textperiodcentered"),
    ):
        m = re.search(pat, raw_line)
        if m:
            issues.append(Issue(file, lineno, "ERROR", "no-middle-dot", f"Middle dots are prohibited ({label}).", _snippet(raw_line, m.start())))

    # 2. Em-dashes (check masked line so code/URLs/math are skipped)
    if is_latex:
        em_pat = re.compile(r"—|---|&mdash;|&#8212;")
    else:
        em_pat = re.compile(r"—|&mdash;|&#8212;")
    for m in em_pat.finditer(masked_line):
        issues.append(
            Issue(
                file,
                lineno,
                "ERROR",
                "no-em-dash",
                "Em-dash found in author prose; replace with commas, a colon, parentheses, or a sentence break.",
                _snippet(raw_line, m.start()),
            )
        )

    # 3. Semicolons (outside HTML entities, code, citations, URLs, verbatim quotes)
    no_quotes = _mask_inline_quotes(masked_line)
    # Remove valid HTML entities (&amp; etc., already checked mdash/middot above)
    semi_scan = re.sub(r"&[a-zA-Z0-9#]+;", lambda m: " " * len(m.group(0)), no_quotes)
    fn_spans = _footnote_spans(raw_line) if is_latex else []
    for m in re.finditer(r";", semi_scan):
        # Determine if inside parentheses or inline \footnote{...} on this line
        before = semi_scan[: m.start()]
        in_parens = before.count("(") > before.count(")")
        in_inline_fn = any(lo <= m.start() <= hi for lo, hi in fn_spans)
        if is_note_or_caption or is_table_row or in_parens or in_inline_fn:
            issues.append(
                Issue(
                    file,
                    lineno,
                    "WARN",
                    "paren-or-note-semicolon",
                    "Semicolon inside parenthetical/note/table; check whether a period or comma reads cleaner.",
                    _snippet(raw_line, m.start()),
                )
            )
        else:
            issues.append(
                Issue(
                    file,
                    lineno,
                    "ERROR",
                    "no-prose-semicolon",
                    "Semicolon in running prose; split into two crisp sentences.",
                    _snippet(raw_line, m.start()),
                )
            )

    # 4. Emphasis attribution
    m_emph = re.search(r"\(emphasis added\)", masked_line, re.IGNORECASE)
    if m_emph:
        issues.append(
            Issue(
                file,
                lineno,
                "ERROR",
                "emphasis-attribution",
                "Use '(emphasis ours)' rather than '(emphasis added)' for author-italicised excerpts.",
                _snippet(raw_line, m_emph.start()),
            )
        )

    # 5. ASCII minus before pp (e.g., "-0.8 pp" instead of "−0.8 pp")
    for m in re.finditer(r"(?<![A-Za-z0-9\\])-(\d+(?:\.\d+)?)\s*(?:~|\u00a0)?pp\b", masked_line):
        issues.append(
            Issue(
                file,
                lineno,
                "ERROR",
                "ascii-minus-pp",
                f"Use true Unicode minus '−{m.group(1)} pp' instead of ASCII hyphen '-{m.group(1)} pp'.",
                _snippet(raw_line, m.start()),
            )
        )

    # 6. British English spelling (on quote-masked prose)
    for m in US_SPELLING_RE.finditer(no_quotes):
        word = m.group(1)
        uk = US_TO_UK[word.lower()]
        # Avoid flagging 'center' inside LaTeX \begin{center} / \end{center}
        if is_latex and word.lower() == "center" and re.search(r"\\(?:begin|end)\{center\}", raw_line):
            continue
        issues.append(
            Issue(
                file,
                lineno,
                "ERROR",
                "en-gb-spelling",
                f"US spelling '{word}' found; use British English '{uk}'.",
                _snippet(raw_line, m.start()),
            )
        )
    for m in LICENSE_NOUN_RE.finditer(no_quotes):
        issues.append(
            Issue(
                file,
                lineno,
                "ERROR",
                "en-gb-spelling",
                f"Use British noun spelling 'licence' instead of '{m.group(0)}'.",
                _snippet(raw_line, m.start()),
            )
        )
    for m in PROGRAM_GRANTS_RE.finditer(no_quotes):
        issues.append(
            Issue(
                file,
                lineno,
                "ERROR",
                "en-gb-spelling",
                f"Use British spelling 'programme' in '{m.group(0)}'.",
                _snippet(raw_line, m.start()),
            )
        )

    # 7. Internal leaks
    for pat, msg in INTERNAL_LEAK_PATTERNS:
        m = pat.search(masked_line)
        if m:
            issues.append(Issue(file, lineno, "ERROR", "internal-leak", msg, _snippet(raw_line, m.start())))

    # 8. Banned slop / boosters (quote-masked)
    for pat, sev, msg in BANNED_SLOP_PATTERNS:
        m = pat.search(no_quotes)
        if m:
            issues.append(Issue(file, lineno, sev, "banned-slop", msg, _snippet(raw_line, m.start())))

    # 9. Domain metaphor & register advisories (skip Appendix E prompt-framing disclosure for 'staffing')
    for pat, msg in DOMAIN_METAPHOR_PATTERNS:
        m = pat.search(no_quotes)
        if m:
            if "staffing" in msg and "staffing reviewer" in raw_line.lower():
                continue
            issues.append(Issue(file, lineno, "WARN", "domain-register", msg, _snippet(raw_line, m.start())))

    # 10. Shortened model nicknames
    if not is_table_row:
        for pat, msg in NICKNAME_PATTERNS:
            m = pat.search(no_quotes)
            if m:
                issues.append(Issue(file, lineno, "WARN", "canonical-model-name", msg, _snippet(raw_line, m.start())))

    # 11. Oxford comma hint
    if not is_table_row:
        for m in OXFORD_COMMA_RE.finditer(no_quotes):
            # Ignore numeric citations like "2026, 2027, and" or author lists "A, B, and C" in footnotes
            seg = m.group(0)
            if re.search(r"\b(?:19|20)\d{2}\b", seg):
                continue
            issues.append(
                Issue(
                    file,
                    lineno,
                    "WARN",
                    "oxford-comma",
                    f"Possible serial (Oxford) comma: '{seg}'. HTAIL style omits serial commas unless required for clarity.",
                    _snippet(raw_line, m.start()),
                )
            )

    # 12. Contrastive negation hint
    for m in CONTRASTIVE_HINT_RE.finditer(no_quotes):
        issues.append(
            Issue(
                file,
                lineno,
                "WARN",
                "contrastive-negation",
                "Possible 'not X, but Y' contrastive negation; verify affirmative phrasing wouldn't read sharper.",
                _snippet(raw_line, m.start()),
            )
        )

    # 13. Soft preference: natural negative absence phrasing over determiner 'zero <noun>'
    for m in ZERO_ABSENCE_RE.finditer(no_quotes):
        phrase = re.sub(r"\s+", " ", m.group(0))
        issues.append(
            Issue(
                file,
                lineno,
                "WARN",
                "prefer-negative-absence",
                f"Soft preference: consider negative absence phrasing ('no ...', 'none of ...', 'without ...', 'was not observed') instead of '{phrase}' (valid construct; advisory only).",
                _snippet(raw_line, m.start()),
            )
        )


def _check_parenthetical_density(
    file: str,
    lineno: int,
    text: str,
    issues: list[Issue],
    *,
    is_note_or_caption: bool = False,
) -> tuple[int, int]:
    """Return (paren_count, word_count) and emit warning if paragraph exceeds threshold."""
    # Remove markdown link targets, code spans, HTML tags, citations
    s = re.sub(r"`[^`]*`", " ", text)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\code\{[^}]*\}", " ", s)
    s = re.sub(r"\\texttt\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:cite[pt]?|ref|label|footnote|shortstack)\b(?:\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\[\^[^\]]+\]|\[@[^\]]+\]", " ", s)
    # Count parenthetical groups (excluding single-letter/table refs like (a) or (1) if trivial? Count all balanced (...))
    parens = len(re.findall(r"\([^()]{3,}\)", s))
    words = len(re.findall(r"\b\w+\b", s))
    limit = 6 if is_note_or_caption else 4
    if parens >= limit:
        kind = "Footnote/caption" if is_note_or_caption else "Paragraph"
        issues.append(
            Issue(
                file,
                lineno,
                "WARN",
                "parenthetical-density",
                f"{kind} stacks {parens} parenthetical clauses in {words} words; consider promoting key qualifiers or moving details to notes/appendices.",
                _snippet(text),
            )
        )
    return parens, words


# ---------------------------------------------------------------------------
# Markdown / Astro Linter
# ---------------------------------------------------------------------------


def lint_markdown(path: Path) -> tuple[list[Issue], dict]:
    issues: list[Issue] = []
    file = str(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    in_frontmatter = False
    in_code = False
    code_fence = ""
    in_quote = False
    in_figure = False
    in_details_summary = False
    in_appendix = False
    in_footnote_section = False

    headings: list[tuple[int, int, str]] = []  # (line, depth, text)
    bullet_period = 0
    bullet_colon = 0
    main_parens = 0
    main_words = 0

    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()

        # Front matter
        if idx == 1 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
                continue
            # Check title, subtitle, abstract, funding in front matter
            m_fm = re.match(r"^(title|subtitle|abstract|funding)\s*:\s*(.*)$", stripped)
            if m_fm:
                key, val = m_fm.group(1), m_fm.group(2).strip().strip('"').strip("'")
                if key == "title":
                    if val.endswith("."):
                        issues.append(Issue(file, idx, "ERROR", "heading-trailing-period", "Front-matter title must not end with a full stop.", val))
                    if "—" in val:
                        issues.append(Issue(file, idx, "ERROR", "no-em-dash", "Em-dash in front-matter title.", val))
                    if "·" in val:
                        issues.append(Issue(file, idx, "ERROR", "no-middle-dot", "Middle dot in front-matter title.", val))
                else:
                    masked = _check_math_spans(file, idx, val, issues)
                    _check_prose_text(file, idx, val, masked, issues, is_note_or_caption=(key == "abstract"))
            continue

        # Fenced code blocks
        m_fence = re.match(r"^(`{3,}|~{3,})", stripped)
        if m_fence:
            if not in_code:
                in_code = True
                code_fence = m_fence.group(1)[0]
            elif stripped.startswith(code_fence * 3):
                in_code = False
                code_fence = ""
            continue
        if in_code:
            continue

        # Directive blocks: :::quote, :::figure, :::fold, :::aside
        if re.match(r"^:::quote\b", stripped):
            in_quote = True
            if "·" in raw:
                issues.append(Issue(file, idx, "ERROR", "no-middle-dot", "Middle dot in :::quote header metadata.", _snippet(raw)))
            if "—" in raw:
                issues.append(Issue(file, idx, "ERROR", "no-em-dash", "Em-dash in :::quote header metadata.", _snippet(raw)))
            continue
        if in_quote:
            if stripped == ":::":
                in_quote = False
            continue

        if re.match(r"^:::figure\b", stripped):
            in_figure = True
            m_alt = re.search(r'alt="([^"]*)"', raw)
            if not m_alt:
                issues.append(Issue(file, idx, "ERROR", "missing-alt-text", ":::figure directive is missing an alt=\"...\" attribute.", _snippet(raw)))
            else:
                alt = m_alt.group(1)
                if not alt.strip():
                    issues.append(Issue(file, idx, "WARN", "missing-alt-text", ":::figure has empty alt text (only permitted for purely decorative images).", _snippet(raw)))
                # Check ASCII hyphens used for negative numbers or numeric ranges inside alt
                for m_h in re.finditer(r"(?<![A-Za-z0-9])-(\d+(?:\.\d+)?)\s*pp\b|\b\d+(?:\.\d+)?%-\d+(?:\.\d+)?%", alt):
                    issues.append(
                        Issue(
                            file,
                            idx,
                            "ERROR",
                            "alt-ascii-hyphen",
                            f"ASCII hyphen in figure alt text ('{m_h.group(0)}'); use Unicode minus '−' or en-dash '–'.",
                            _snippet(alt, m_h.start()),
                        )
                    )
                if "·" in alt:
                    issues.append(Issue(file, idx, "ERROR", "no-middle-dot", "Middle dot inside figure alt text.", _snippet(alt)))
                if "—" in alt:
                    issues.append(Issue(file, idx, "ERROR", "no-em-dash", "Em-dash inside figure alt text.", _snippet(alt)))
                for m_z in ZERO_ABSENCE_RE.finditer(alt):
                    phrase = re.sub(r"\s+", " ", m_z.group(0))
                    issues.append(
                        Issue(
                            file,
                            idx,
                            "WARN",
                            "prefer-negative-absence",
                            f"Soft preference: consider negative absence phrasing ('no ...', 'without ...') instead of '{phrase}' in figure alt text (advisory only).",
                            _snippet(alt, m_z.start()),
                        )
                    )
            continue
        if in_figure:
            if stripped == ":::":
                in_figure = False
                continue
            # Inside figure caption
            masked = _mask_md_inline(file, idx, raw, issues)
            _check_prose_text(file, idx, raw, masked, issues, is_note_or_caption=True)
            _check_parenthetical_density(file, idx, raw, issues, is_note_or_caption=True)
            continue

        if stripped.startswith(":::") or stripped in ("<details>", "</details>", "---"):
            continue
        if stripped.startswith("<summary>"):
            in_details_summary = False
            summary_text = re.sub(r"<[^>]+>", "", stripped)
            if "·" in summary_text:
                issues.append(Issue(file, idx, "ERROR", "no-middle-dot", "Middle dot in <summary>.", summary_text))
            if "—" in summary_text:
                issues.append(Issue(file, idx, "ERROR", "no-em-dash", "Em-dash in <summary>.", summary_text))
            continue

        # Standalone Markdown blockquotes (> ...) — treat as verbatim quote body except speaker header line
        if stripped.startswith(">"):
            if re.match(r"^>\s*\*\*[^*]+\*\*\s*\*\([^)]*\)\*\s*:?\s*$", stripped):
                if "·" in raw:
                    issues.append(Issue(file, idx, "ERROR", "no-middle-dot", "Middle dot in blockquote speaker header.", _snippet(raw)))
                if "—" in raw:
                    issues.append(Issue(file, idx, "ERROR", "no-em-dash", "Em-dash in blockquote speaker header.", _snippet(raw)))
            continue

        # Image markdown ![alt](src)
        if re.match(r"^!\[[^\]]*\]\([^)]+\)\s*$", stripped):
            m_img = re.match(r"^!\[([^\]]*)\]", stripped)
            if m_img:
                alt = m_img.group(1)
                for m_h in re.finditer(r"(?<![A-Za-z0-9])-(\d+(?:\.\d+)?)\s*pp\b|\b\d+(?:\.\d+)?%-\d+(?:\.\d+)?%", alt):
                    issues.append(Issue(file, idx, "ERROR", "alt-ascii-hyphen", f"ASCII hyphen in image alt text ('{m_h.group(0)}').", _snippet(alt)))
                for m_z in ZERO_ABSENCE_RE.finditer(alt):
                    phrase = re.sub(r"\s+", " ", m_z.group(0))
                    issues.append(
                        Issue(
                            file,
                            idx,
                            "WARN",
                            "prefer-negative-absence",
                            f"Soft preference: consider negative absence phrasing ('no ...', 'without ...') instead of '{phrase}' in image alt text (advisory only).",
                            _snippet(alt, m_z.start()),
                        )
                    )
            continue

        # Headings
        m_hd = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m_hd:
            depth = len(m_hd.group(1))
            htext = m_hd.group(2).strip()
            headings.append((idx, depth, htext))
            if depth == 2 and re.search(r"technical appendices", htext, re.I):
                in_appendix = True
            if depth == 2 and re.search(r"^footnotes$", htext, re.I):
                in_footnote_section = True
            if depth >= 2:
                _check_heading(file, idx, htext, issues)
            masked = _mask_md_inline(file, idx, htext, issues)
            _check_prose_text(file, idx, htext, masked, issues)
            continue

        if not stripped:
            continue

        # Table rows
        is_table = stripped.startswith("|") and stripped.endswith("|")
        if is_table:
            if re.match(r"^\|[\s:|-]+\|$", stripped):
                continue
            masked = _mask_md_inline(file, idx, raw, issues)
            _check_prose_text(file, idx, raw, masked, issues, is_table_row=True)
            continue

        # Footnote definitions [^id]: ...
        is_fn = bool(re.match(r"^\[\^[^\]]+\]:", stripped)) or in_footnote_section
        is_caption = bool(re.match(r"^\*(?:Figure|Table)\s+[A-Z]?\d+[:.]", stripped))

        # Bold-lead-in bullets consistency tracking (main body only)
        if not in_appendix and not is_fn:
            m_bul = re.match(r"^(?:[-*]|\d+\.)\s+\*\*[^*]+([.:])\*\*", stripped)
            if m_bul:
                if m_bul.group(1) == ".":
                    bullet_period += 1
                else:
                    bullet_colon += 1

        # Regular author prose / caption / footnote
        # Skip standalone metadata header lines in docs/blog_post.md (Authors / Resources / Technical Report)
        if idx <= 8 and re.match(r"^\*\*(?:Authors|Resources|Technical Report)\*\*?", stripped):
            if "·" in raw:
                issues.append(Issue(file, idx, "ERROR", "no-middle-dot", "Middle dot in document header.", _snippet(raw)))
            if "—" in raw:
                issues.append(Issue(file, idx, "ERROR", "no-em-dash", "Em-dash in document header (use comma or newline).", _snippet(raw)))
            continue

        masked = _mask_md_inline(file, idx, raw, issues)
        _check_prose_text(
            file,
            idx,
            raw,
            masked,
            issues,
            is_note_or_caption=(is_fn or is_caption),
        )
        p_cnt, w_cnt = _check_parenthetical_density(
            file,
            idx,
            raw,
            issues,
            is_note_or_caption=(is_fn or is_caption),
        )
        if not in_appendix and not is_fn and not is_caption:
            main_parens += p_cnt
            main_words += w_cnt

    # Document-level advisories
    if main_words > 200 and main_parens > 0:
        ratio = main_words / main_parens
        if ratio < 50.0:
            issues.append(
                Issue(
                    file,
                    1,
                    "WARN",
                    "parenthetical-density-doc",
                    f"Main-body parenthetical density is 1 per {ratio:.1f} words ({main_parens} parentheticals in {main_words} words; target is sparser than 1 per 50 words).",
                )
            )
    if bullet_period > 0 and bullet_colon > 0:
        issues.append(
            Issue(
                file,
                1,
                "WARN",
                "bullet-punctuation-mix",
                f"Bold-lead-in bullets mix trailing periods ({bullet_period} ending '.**') and colons ({bullet_colon} ending ':**') across lists.",
            )
        )

    meta = {
        "headings": headings,
        "main_parens": main_parens,
        "main_words": main_words,
        "under_inform_count": len(re.findall(r"\bunder-inform\w*", "\n".join(lines), re.I)),
        "withhold_count": len(re.findall(r"\bwithhold\w*|\bwithheld\b", "\n".join(lines), re.I)),
        "has_figure_a1": bool(re.search(r"\bFigure\s+A1\b", "\n".join(lines))),
        "has_figure_8": bool(re.search(r"\bFigure\s+8\b", "\n".join(lines))),
        "has_footnote_number_ref": bool(re.search(r"\bFootnote\s+\d+\b", "\n".join(lines))),
    }
    return issues, meta


def _mask_md_inline(file: str, lineno: int, line: str, issues: list[Issue]) -> str:
    # 1. Mask inline code spans `...`
    s = re.sub(r"`[^`]+`", lambda m: " " * len(m.group(0)), line)
    # 2. Audit & mask math spans $...$
    s = _check_math_spans(file, lineno, s, issues)
    # 3. Mask Markdown link URLs while preserving link text
    s = re.sub(r"(\[[^\]]*\])\(([^)]+)\)", lambda m: m.group(1) + " " * (len(m.group(2)) + 2), s)
    # 4. Mask bare URLs
    s = re.sub(r"https?://\S+", lambda m: " " * len(m.group(0)), s)
    # 5. Mask BibTeX citations [@a; @b] and footnote refs [^id]
    s = re.sub(r"\[@[^\]]+\]", lambda m: " " * len(m.group(0)), s)
    s = re.sub(r"\[\^[^\]]+\]:?", lambda m: " " * len(m.group(0)), s)
    # 6. Mask HTML tags
    s = re.sub(r"<[^>]+>", lambda m: " " * len(m.group(0)), s)
    return s


# ---------------------------------------------------------------------------
# LaTeX Linter
# ---------------------------------------------------------------------------

LATEX_SKIP_ENVS = {
    "modelquote",
    "promptbox",
    "titledprompt",
    "shellbox",
    "verbatim",
    "lstlisting",
    "equation",
    "equation*",
    "align",
    "align*",
}


def _find_balanced_brace(s: str, open_idx: int) -> int:
    """Given index of '{', return index of matching unescaped '}', or -1."""
    depth = 0
    i = open_idx
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _mask_latex_cmd_with_arg(s: str, cmds: set[str]) -> str:
    """Replace \\cmd{...} (supporting escaped braces and nesting) with spaces."""
    pat = re.compile(r"\\(" + "|".join(sorted(cmds, key=len, reverse=True)) + r")\*?(?:\[[^\]]*\])?\{")
    chars = list(s)
    pos = 0
    while True:
        m = pat.search(s, pos)
        if not m:
            break
        open_idx = m.end() - 1
        close_idx = _find_balanced_brace(s, open_idx)
        if close_idx == -1:
            pos = m.end()
            continue
        for k in range(m.start(), close_idx + 1):
            chars[k] = " "
        pos = close_idx + 1
    return "".join(chars)


def _footnote_spans(s: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        idx = s.find(r"\footnote{", pos)
        if idx == -1:
            break
        open_idx = idx + len(r"\footnote")
        close_idx = _find_balanced_brace(s, open_idx)
        if close_idx == -1:
            break
        spans.append((open_idx, close_idx))
        pos = close_idx + 1
    return spans


def _mask_latex_inline(file: str, lineno: int, line: str, issues: list[Issue]) -> str:
    # Strip unescaped % comments
    s = re.sub(r"(?<!\\)%.*$", "", line)
    # Mask code, URL, reference, colour, and layout commands with balanced braces
    s = _mask_latex_cmd_with_arg(
        s,
        {
            "code", "texttt", "nolinkurl", "url", "label", "ref", "eqref",
            "citep", "citet", "cite", "bibliographystyle", "bibliography",
            "usepackage", "documentclass", "htailfigure", "includegraphics",
            "setlength", "renewcommand", "newcommand", "hypersetup",
            "color", "rule", "vspace", "hspace", "addlinespace",
        },
    )
    # Mask only the first (URL) argument of \href{url}{text} and \textcolor{col}{text}
    for two_arg in ("href", "textcolor"):
        pat = re.compile(r"\\" + two_arg + r"\{")
        chars = list(s)
        pos = 0
        while True:
            m = pat.search(s, pos)
            if not m:
                break
            close_1 = _find_balanced_brace(s, m.end() - 1)
            if close_1 == -1:
                pos = m.end()
                continue
            for k in range(m.start(), close_1 + 1):
                chars[k] = " "
            pos = close_1 + 1
        s = "".join(chars)
    # Audit & mask inline math $...$
    s = _check_math_spans(file, lineno, s, issues)
    # Mask remaining LaTeX control sequences (e.g., \textbf, \emph, \footnote, \htailtrack)
    s = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", lambda m: " " * len(m.group(0)), s)
    return s


def lint_latex(path: Path) -> tuple[list[Issue], dict]:
    issues: list[Issue] = []
    file = str(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    in_doc = False
    skip_env: str | None = None
    in_abstract = False
    in_table = False
    in_appendix = False
    headings: list[tuple[int, int, str]] = []
    main_parens = 0
    main_words = 0

    for idx, raw in enumerate(lines, start=1):
        no_comment = re.sub(r"(?<!\\)%.*$", "", raw)
        stripped = no_comment.strip()

        if not in_doc:
            if re.search(r"\\pdfoutput\s*=\s*1", no_comment):
                issues.append(Issue(file, idx, "ERROR", "latex-pdfoutput-xelatex", "Never use \\pdfoutput=1 with XeLaTeX (declare compiler in 00README.json).", _snippet(raw)))
            if re.search(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\b(?:amssymb|amsfonts)\b", no_comment):
                issues.append(Issue(file, idx, "ERROR", "latex-incompatible-math-pkg", "Do not load amssymb/amsfonts alongside unicode-math in htail-report.", _snippet(raw)))
            if "·" in no_comment or r"\textperiodcentered" in no_comment:
                issues.append(Issue(file, idx, "ERROR", "no-middle-dot", "Middle dot found in LaTeX preamble/masthead.", _snippet(raw)))

            if r"\htailAbstract{" in no_comment:
                in_abstract = True
                continue
            if in_abstract:
                if stripped == "}":
                    in_abstract = False
                    continue
                masked = _mask_latex_inline(file, idx, no_comment, issues)
                _check_prose_text(file, idx, no_comment, masked, issues, is_latex=True, is_note_or_caption=True)
                _check_parenthetical_density(file, idx, no_comment, issues, is_note_or_caption=True)
                continue

            m_sub = re.search(r"\\htailSubtitle\{(.+)\}", no_comment)
            if m_sub:
                sub = m_sub.group(1)
                masked = _mask_latex_inline(file, idx, sub, issues)
                _check_prose_text(file, idx, sub, masked, issues, is_latex=True)

            m_title = re.search(r"\\title\{(.+)\}", no_comment)
            if m_title and m_title.group(1).strip().endswith("."):
                issues.append(Issue(file, idx, "ERROR", "heading-trailing-period", "LaTeX \\title must not end with a full stop.", m_title.group(1)))

            if r"\begin{document}" in no_comment:
                in_doc = True
            continue

        # Environment skipping
        if skip_env is not None:
            if rf"\end{{{skip_env}}}" in no_comment:
                skip_env = None
            continue

        m_beg = re.search(r"\\begin\{([a-zA-Z*]+)\}", no_comment)
        if m_beg:
            env = m_beg.group(1)
            if env in LATEX_SKIP_ENVS:
                if env == "modelquote":
                    # Check speaker & meta args on the \begin{modelquote} line
                    if "·" in no_comment or r"\textperiodcentered" in no_comment:
                        issues.append(Issue(file, idx, "ERROR", "no-middle-dot", "Middle dot in modelquote header.", _snippet(raw)))
                    if "—" in no_comment or "---" in no_comment:
                        issues.append(Issue(file, idx, "ERROR", "no-em-dash", "Em-dash in modelquote header.", _snippet(raw)))
                if rf"\end{{{env}}}" not in no_comment:
                    skip_env = env
                continue
            if env in ("tabular", "tabularx", "longtable"):
                in_table = True
                continue

        if re.search(r"\\end\{(?:tabular|tabularx|longtable)\}", no_comment):
            in_table = False
            continue

        if r"\appendix" in no_comment:
            in_appendix = True
            continue

        if not stripped or stripped in (r"\begin{itemize}", r"\end{itemize}", r"\begin{enumerate}", r"\end{enumerate}", r"\begin{figure}[htbp]", r"\end{figure}", r"\begin{table}[htbp]", r"\end{table}", r"\end{document}", r"\makehtailtitle"):
            continue

        # Section / subsection headings
        m_sec = re.match(r"^\\(sub)?section\*?\{(.+)\}\s*$", stripped)
        if m_sec:
            depth = 3 if m_sec.group(1) else 2
            htext = m_sec.group(2)
            headings.append((idx, depth, htext))
            _check_heading(file, idx, htext, issues)
            masked = _mask_latex_inline(file, idx, htext, issues)
            _check_prose_text(file, idx, htext, masked, issues, is_latex=True)
            continue

        # Skip pure formatting/layout lines
        if re.match(r"^\\(?:label|centering|htailfigure|vspace|hspace|rule|addcontentsline|bibliographystyle|bibliography|renewcommand|makeat|@addtoreset|begin\{center\}|end\{center\}|begin\{techbox\}|end\{techbox\}|setlength|scriptsize|footnotesize|small|toprule|midrule|bottomrule|addlinespace)", stripped):
            continue

        is_caption = stripped.startswith(r"\caption{")
        masked = _mask_latex_inline(file, idx, no_comment, issues)

        # Check straight ASCII quotes in LaTeX prose (outside code/tables/commands)
        if not in_table and '"' in masked:
            issues.append(
                Issue(
                    file,
                    idx,
                    "WARN",
                    "latex-straight-quotes",
                    "Straight ASCII double quote '\"' in LaTeX prose; use ``...'' typographic quotes.",
                    _snippet(raw),
                )
            )

        _check_prose_text(
            file,
            idx,
            no_comment,
            masked,
            issues,
            is_latex=True,
            is_note_or_caption=is_caption,
            is_table_row=in_table,
        )
        if not in_table:
            p_cnt, w_cnt = _check_parenthetical_density(
                file,
                idx,
                no_comment,
                issues,
                is_note_or_caption=is_caption,
            )
            if not in_appendix and not is_caption:
                main_parens += p_cnt
                main_words += w_cnt

    if main_words > 200 and main_parens > 0:
        ratio = main_words / main_parens
        if ratio < 50.0:
            issues.append(
                Issue(
                    file,
                    1,
                    "WARN",
                    "parenthetical-density-doc",
                    f"Main-body parenthetical density is 1 per {ratio:.1f} words ({main_parens} parentheticals in {main_words} words).",
                )
            )

    return issues, {"headings": headings, "main_parens": main_parens, "main_words": main_words}


# ---------------------------------------------------------------------------
# SVG Figure Hygiene Checker
# ---------------------------------------------------------------------------


def lint_svg_dir(svg_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    if not svg_dir.is_dir():
        return issues
    for svg_path in sorted(svg_dir.glob("*.svg")):
        file = str(svg_path)
        try:
            tree = ET.parse(svg_path)
        except ET.ParseError as exc:
            issues.append(Issue(file, 1, "ERROR", "svg-parse-error", f"Could not parse SVG: {exc}"))
            continue
        texts: list[str] = []
        for elem in tree.iter():
            tag = elem.tag.split("}")[-1]
            if tag in ("text", "tspan") and elem.text and elem.text.strip():
                texts.append(elem.text.strip())
        joined = " | ".join(texts)
        for t in texts:
            if "·" in t:
                issues.append(Issue(file, 1, "ERROR", "svg-no-middle-dot", "Middle dot found in SVG figure text.", t))
            if "—" in t:
                issues.append(Issue(file, 1, "ERROR", "svg-no-em-dash", "Em-dash found in SVG figure text.", t))
            if ";" in t:
                issues.append(Issue(file, 1, "ERROR", "svg-no-semicolon", "Semicolon found in SVG figure text.", t))
            for m in US_SPELLING_RE.finditer(t):
                w = m.group(1)
                issues.append(Issue(file, 1, "ERROR", "svg-en-gb-spelling", f"US spelling '{w}' in SVG text; use '{US_TO_UK[w.lower()]}'.", t))
            for pat, msg in (
                (re.compile(r"\bwithhold\w*|\bwithheld\b", re.I), "Use 'under-informing' / 'under-informed' instead of 'withholding' in SVG."),
                (re.compile(r"\bhir(?:e|es|ed|ing)\b|\bstaffing\b|\bsurvival\b|\bv2\.1\b", re.I), "Banned study jargon/metaphor in SVG."),
            ):
                if pat.search(t):
                    issues.append(Issue(file, 1, "ERROR", "svg-terminology", msg, t))
        _ = joined
    return issues


# ---------------------------------------------------------------------------
# Cross-File Parity Checker (index.md vs docs/blog_post.md)
# ---------------------------------------------------------------------------


def check_parity(primary_path: Path, primary_meta: dict, other_path: Path, other_meta: dict) -> list[Issue]:
    issues: list[Issue] = []
    p_name = str(primary_path)
    o_name = str(other_path)

    p_h23 = [(ln, d, t) for (ln, d, t) in primary_meta["headings"] if d in (2, 3)]
    o_h23 = [(ln, d, t) for (ln, d, t) in other_meta["headings"] if d in (2, 3) and not t.lower() == "footnotes"]

    if len(p_h23) != len(o_h23):
        issues.append(
            Issue(
                o_name,
                1,
                "WARN",
                "parity-heading-count",
                f"Section/subsection count differs from {p_name}: {len(o_h23)} vs. {len(p_h23)}.",
            )
        )

    for (pln, pd, pt), (oln, od, ot) in zip(p_h23, o_h23):
        if pd == od and pt.lower() == ot.lower() and pt != ot:
            issues.append(
                Issue(
                    o_name,
                    oln,
                    "WARN",
                    "parity-heading-case",
                    f"Heading casing differs from {p_name}:{pln} ('{ot}' vs. '{pt}').",
                )
            )
        elif pt.lower() != ot.lower():
            issues.append(
                Issue(
                    o_name,
                    oln,
                    "WARN",
                    "parity-heading-text",
                    f"Heading text differs from {p_name}:{pln} ('{ot}' vs. '{pt}').",
                )
            )

    if primary_meta.get("has_figure_8") and other_meta.get("has_figure_a1"):
        issues.append(
            Issue(
                o_name,
                1,
                "WARN",
                "parity-figure-numbering",
                f"Appendix figure numbered 'Figure A1' in {o_name} vs. auto-numbered 'Figure 8' in {p_name}.",
            )
        )

    if other_meta.get("has_footnote_number_ref") and not primary_meta.get("has_footnote_number_ref"):
        issues.append(
            Issue(
                o_name,
                1,
                "WARN",
                "parity-footnote-ref",
                f"Explicit numeric 'Footnote N' reference appears in {o_name} but not in {p_name} (where margin notes interleave footnotes and citations).",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# CLI & Formatting
# ---------------------------------------------------------------------------


def format_text(issues: list[Issue]) -> str:
    if not issues:
        return "HTAIL prose lint: 0 errors, 0 warnings."
    errors = [i for i in issues if i.severity == "ERROR"]
    warns = [i for i in issues if i.severity == "WARN"]
    lines = [f"HTAIL prose lint: {len(errors)} error(s), {len(warns)} warning(s)", ""]
    for i in issues:
        loc = f"{i.file}:{i.line}"
        lines.append(f"[{i.severity}] {loc} ({i.rule}): {i.message}")
        if i.snippet:
            lines.append(f"    {i.snippet}")
    return "\n".join(lines)


def format_markdown(issues: list[Issue]) -> str:
    errors = [i for i in issues if i.severity == "ERROR"]
    warns = [i for i in issues if i.severity == "WARN"]
    out = [f"### HTAIL Prose Lint Summary: **{len(errors)} errors**, **{len(warns)} warnings**", ""]
    if not issues:
        out.append("All checked files passed with no errors or warnings.")
        return "\n".join(out)
    out.append("| Severity | Location | Rule | Detail |")
    out.append("| :--- | :--- | :--- | :--- |")
    for i in issues:
        msg = i.message.replace("|", "\\|")
        out.append(f"| `{i.severity}` | `{i.file}:{i.line}` | `{i.rule}` | {msg} |")
    return "\n".join(out)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic HTAIL prose, LaTeX, and SVG house-style linter.")
    parser.add_argument("files", nargs="+", help="Markdown (.md) or LaTeX (.tex) files to lint.")
    parser.add_argument("--compare", type=Path, default=None, help="Companion Markdown file to check parity against the first file.")
    parser.add_argument("--svg-dir", type=Path, default=None, help="Optional directory of SVG figures to audit.")
    parser.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    parser.add_argument("--warn-only", action="store_true", help="Exit 0 even if errors are found.")
    parser.add_argument("--fail-on-warn", action="store_true", help="Exit 1 on warnings as well as errors.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    all_issues: list[Issue] = []
    primary_path: Path | None = None
    primary_meta: dict | None = None

    for fstr in args.files:
        p = Path(fstr)
        if not p.exists():
            all_issues.append(Issue(fstr, 0, "ERROR", "file-not-found", f"File not found: {fstr}"))
            continue
        if p.suffix.lower() == ".tex":
            file_issues, meta = lint_latex(p)
        else:
            file_issues, meta = lint_markdown(p)
        all_issues.extend(file_issues)
        if primary_path is None:
            primary_path, primary_meta = p, meta

    if args.compare is not None:
        if not args.compare.exists():
            all_issues.append(Issue(str(args.compare), 0, "ERROR", "file-not-found", f"Comparison file not found: {args.compare}"))
        else:
            other_issues, other_meta = lint_markdown(args.compare)
            all_issues.extend(other_issues)
            if primary_path is not None and primary_meta is not None and primary_path.suffix.lower() != ".tex":
                all_issues.extend(check_parity(primary_path, primary_meta, args.compare, other_meta))

    if args.svg_dir is not None:
        all_issues.extend(lint_svg_dir(args.svg_dir))

    if args.format == "json":
        print(json.dumps([asdict(i) for i in all_issues], indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        print(format_markdown(all_issues))
    else:
        print(format_text(all_issues))

    if args.warn_only:
        return 0
    has_errors = any(i.severity == "ERROR" for i in all_issues)
    has_warns = any(i.severity == "WARN" for i in all_issues)
    if has_errors or (args.fail_on_warn and has_warns):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
