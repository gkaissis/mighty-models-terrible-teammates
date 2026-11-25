"""Generate publication figures (SVG + 300-DPI PNG) and downloadable JSON datasets.

All numeric values are loaded programmatically from the frozen summary JSONs and raw records:
  - data/v2/analysis/meta_v2_1.json
  - data/v2/exp1_dropout/exp1_summary.json
  - data/v2/exp1_dropout/gate_report.json
  - data/v2/exp1_dropout/results.jsonl
  - data/v2/exp2_info_flip/exp2_summary.json
  - data/v2/episodes/triage__s30[1-4]__*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import xml.etree.ElementTree as ET
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parent.parent
META_PATH = REPO_ROOT / "data/v2/analysis/meta_v2_1.json"
EXP1_PATH = REPO_ROOT / "data/v2/exp1_dropout/exp1_summary.json"
EXP1_RESULTS_PATH = REPO_ROOT / "data/v2/exp1_dropout/results.jsonl"
GATE_PATH = REPO_ROOT / "data/v2/exp1_dropout/gate_report.json"
EXP2_PATH = REPO_ROOT / "data/v2/exp2_info_flip/exp2_summary.json"
EPISODES_DIR = REPO_ROOT / "data/v2/episodes"

FIG_DIR = REPO_ROOT / "docs/assets/figures"
DATA_DIR = REPO_ROOT / "docs/assets/data"
SITE_FIG_DIR = REPO_ROOT / "site/src/figures"
SITE_DATA_DIR = REPO_ROOT / "site/public/data"
PDF_OUTPUT_DIR: Path | None = None

# House editorial palette matching article-kit (warm ground #F1EADC, ink #000000, dark red/teal/forest/ochre)
BG_WHITE = "#F1EADC"
TEXT_DARK = "#000000"
TEXT_BODY = "#4B433B"
TEXT_TICK = "#5E564D"
MUTED_DARK = "#5E564D"
MUTED = "#6A6158"
MUTED_LIGHT = "#D5C9B4"
GREY_BAR_DARK = "#9E9487"
BORDER = "#E0D6C3"
ROW_TINT = "#E8DEC9"

TEAL = "#0E3A43"
TEAL_DEEP = "#08252B"
TEAL_SOFT = "#3B6B73"
SLATE_BLUE = "#1B3A18"
VERMILION = "#680F0D"
VERMILION_DEEP = "#4A0908"
VERMILION_SOFT = "#9A3F3B"
AMBER = "#A85C19"
TILE_CLEAN = "#F9F4EA"
TILE_MINOR = "#DFD4BE"

# Honest 0%-100% CVD-safe sequential alert-pop ramp (house dark red at 0% -> calm muted teal at 100%)
ALERT_SEQ_COLORS = [
    (0.00, "#680F0D"),
    (0.25, "#9A372B"),
    (0.50, "#C77542"),
    (0.70, "#DFB484"),
    (0.85, "#F1EADC"),
    (0.94, "#D7E1DD"),
    (1.00, "#BDD1CE"),
]
ALERT_SEQ_CMAP = LinearSegmentedColormap.from_list("cvd_alert_seq", ALERT_SEQ_COLORS)

# Canonical full model names used consistently across all charts, tables and prose
MODEL_FULL: dict[str, str] = {
    "astra62": "GPT-6.2 Astra",
    "astra60": "GPT-6 Astra",
    "neon4": "Gemini 4 Neon",
    "opus55": "Claude Opus 5.5",
    "kimi31": "Kimi K3.1 Pro",
    "muse": "Meta Muse Wonder",
    "qwen4": "Qwen4 Max",
    "hydrogen": "Gemini 4 Hydrogen",
    "deepseek_flash": "DeepSeek V4.1 Flash",
    "haiku55": "Claude Haiku 5.5",
    "gpt6_luna": "GPT-6 Luna",
}

MODEL_TWO_LINE: dict[str, str] = {
    "astra62": "GPT-6.2\nAstra",
    "astra60": "GPT-6\nAstra",
    "neon4": "Gemini 4\nNeon",
    "opus55": "Claude\nOpus 5.5",
    "kimi31": "Kimi\nK3.1 Pro",
    "muse": "Meta Muse\nWonder",
    "qwen4": "Qwen4\nMax",
}

RECEIVER_LADDER_LABEL: dict[str, str] = {
    "astra62": "GPT-6.2 Astra (62)",
    "neon4": "Gemini 4 Neon (61)",
    "opus55": "Claude Opus 5.5 (59)",
    "astra60": "GPT-6 Astra (57)",
    "hydrogen": "Gemini 4 Hydrogen (low, 27)",
    "deepseek_flash": "DeepSeek V4.1 Flash (off, 19)",
    "haiku55": "Claude Haiku 5.5 (low, 19)",
    "gpt6_luna": "GPT-6 Luna (low, unscored)",
}

RECEIVER_COL_LABEL: dict[str, str] = {
    "astra62": "GPT-6.2 Astra\nIndex 62",
    "neon4": "Gemini 4 Neon\nIndex 61",
    "opus55": "Claude Opus 5.5\nIndex 59",
    "astra60": "GPT-6 Astra\nIndex 57",
    "hydrogen": "Gemini 4\nHydrogen\nIndex 27 (low)",
    "deepseek_flash": "DeepSeek\nV4.1 Flash\nIndex 19 (off)",
    "haiku55": "Claude\nHaiku 5.5\nIndex 19 (low)",
    "gpt6_luna": "GPT-6\nLuna\nunscored (low)",
}

MODEL_FAMILY: dict[str, str] = {
    "astra62": "openai",
    "astra60": "openai",
    "gpt6_luna": "openai",
    "neon4": "google",
    "hydrogen": "google",
    "opus55": "anthropic",
    "haiku55": "anthropic",
    "kimi31": "moonshot",
    "muse": "meta",
    "qwen4": "qwen",
    "deepseek_flash": "deepseek",
}

BANNED_FIGURE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bhir(?:e|es|ed|ing)\b", "hiring metaphor"),
    (r"\bstaffing\b", "staffing metaphor"),
    (r"\blabs?\b", "lab metonym"),
    (r"\bsurvival\b", "survival metaphor"),
    (r"\bgat(?:e|es|ed|ekeeper)\b", "internal jargon (gate)"),
    (r"truncations\s*=\s*fail", "truncations=fail jargon"),
    (r"\bnormaliz(?:e|ed|ing)\b", "US spelling (normalise)"),
    (r"\bstandardiz(?:e|ed|ing)\b", "US spelling (standardise)"),
    (r"\bfavor(?:s|ed|ing)?\b", "US spelling (favour)"),
    (r"\bbehavior(?:s|al)?\b", "US spelling (behaviour)"),
    (r"\bpercent\b", "US spelling (percentage)"),
    (r"\btoward\b", "US spelling (towards)"),
)


def apply_runtime_options(theme: str, pdf_dir: Path | None) -> None:
    """Apply optional paper theme palette overrides and PDF-only export target."""
    global BG_WHITE, TILE_CLEAN, ROW_TINT, BORDER, PDF_OUTPUT_DIR
    PDF_OUTPUT_DIR = pdf_dir
    if theme == "paper":
        BG_WHITE = "#FFFFFF"
        TILE_CLEAN = "#F7F5F0"
        ROW_TINT = "#F5F3EE"
        BORDER = "#DDD8CE"


def configure_editorial_style(theme: str = "web", font_dir: Path | None = None) -> None:
    """Apply clean publication rcParams."""
    serif_stack = ["Georgia", "DejaVu Serif"]
    if font_dir is not None:
        for font_file in sorted([*font_dir.glob("*.ttf"), *font_dir.glob("*.otf")]):
            font_manager.fontManager.addfont(str(font_file))
        serif_stack = ["Spectral", "Georgia", "DejaVu Serif"]
    elif theme == "paper":
        serif_stack = ["Spectral", "Georgia", "DejaVu Serif"]

    plt.rcParams.update(
        {
            "figure.facecolor": BG_WHITE,
            "axes.facecolor": BG_WHITE,
            "savefig.facecolor": BG_WHITE,
            "savefig.edgecolor": BG_WHITE,
            "font.family": "serif",
            "font.serif": serif_stack,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.edgecolor": BORDER,
            "axes.linewidth": 0.8,
            "axes.labelcolor": TEXT_BODY,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.2,
            "axes.titlecolor": TEXT_DARK,
            "xtick.color": TEXT_TICK,
            "ytick.color": TEXT_TICK,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "grid.color": BORDER,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "legend.frameon": False,
            "legend.fontsize": 8.2,
            "axes.unicode_minus": False,
        }
    )


def style_axes(ax: plt.Axes, grid_axis: str | None = "x") -> None:
    """Apply minimal spine and grid styling to an axes object."""
    ax.set_facecolor(BG_WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)
    if grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.65, color=BORDER, zorder=0)
    else:
        ax.grid(False)


def _check_text_hygiene(txt: str, ctx: str) -> None:
    if "\u2014" in txt:
        raise ValueError(f"Em-dash prohibited in {ctx}: {txt!r}")
    if "\u00b7" in txt:
        raise ValueError(f"Middle dot prohibited in {ctx}: {txt!r}")
    if ";" in txt:
        raise ValueError(f"Semicolon prohibited in {ctx}: {txt!r}")
    for pat, desc in BANNED_FIGURE_PATTERNS:
        if re.search(pat, txt, flags=re.IGNORECASE):
            raise ValueError(f"Prohibited wording ({desc}) in {ctx}: {txt!r}")


def add_header(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    left_x: float = 0.04,
    title_y: float = 0.96,
    subtitle_y: float = 0.905,
    title_fontsize: float = 13.0,
) -> None:
    """Add crisp left-aligned sentence-case takeaway title and concise subtitle."""
    _check_text_hygiene(title, "title")
    _check_text_hygiene(subtitle, "subtitle")
    fig.text(
        left_x,
        title_y,
        title,
        ha="left",
        va="top",
        fontsize=title_fontsize,
        fontweight="bold",
        color=TEXT_DARK,
    )
    fig.text(
        left_x,
        subtitle_y,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.1,
        color=MUTED,
    )


def assert_no_em_dashes(obj: Any, path: str = "root") -> None:
    """Recursively verify that no em-dash appears in exported data."""
    if isinstance(obj, str):
        if "\u2014" in obj:
            raise ValueError(f"Em-dash found at {path}: {obj!r}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_em_dashes(k, f"{path}.{k}")
            assert_no_em_dashes(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            assert_no_em_dashes(item, f"{path}[{idx}]")


def assert_svg_text_clean(svg_path: Path) -> None:
    """Verify zero em-dashes, semicolons or banned words across every SVG <text> element."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    for elem in root.iter():
        if elem.tag.endswith("}text") or elem.tag == "text":
            txt = "".join(elem.itertext())
            _check_text_hygiene(txt, svg_path.name)


def save_figure_and_data(fig: plt.Figure, stem: str, payload: dict[str, Any]) -> None:
    """Export SVG, 300-DPI PNG and companion JSON (or PDF-only when --pdf-dir is set)."""
    for text_obj in fig.findobj(match=matplotlib.text.Text):
        txt = text_obj.get_text()
        if txt:
            _check_text_hygiene(txt, stem)
    assert_no_em_dashes(payload, stem)

    if PDF_OUTPUT_DIR is not None:
        PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = PDF_OUTPUT_DIR / f"{stem}.pdf"
        fig.savefig(
            pdf_path,
            format="pdf",
            facecolor=BG_WHITE,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(fig)
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    svg_path = FIG_DIR / f"{stem}.svg"
    png_path = FIG_DIR / f"{stem}.png"
    json_path = DATA_DIR / f"{stem}.json"

    plt.rcParams["svg.hashsalt"] = f"mighty-models-v2-1-{stem}"
    fig.savefig(svg_path, format="svg", facecolor=BG_WHITE, metadata={"Date": None})
    fig.savefig(png_path, format="png", dpi=300, facecolor=BG_WHITE)
    plt.close(fig)

    assert_svg_text_clean(svg_path)
    json_text = json.dumps(payload, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8")

    # Mirror SVG & JSON directly into Astro site/ when present
    if SITE_FIG_DIR.parent.is_dir():
        SITE_FIG_DIR.mkdir(parents=True, exist_ok=True)
        SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (SITE_FIG_DIR / f"{stem}.svg").write_bytes(svg_path.read_bytes())
        (SITE_DATA_DIR / f"{stem}.json").write_text(json_text, encoding="utf-8")


def fmt_half_up_3dp(val: float) -> str:
    """Format a float to 3 decimal places using standard half-up rounding."""
    return str(Decimal(str(val)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def luminance(rgba: tuple[float, float, float, float]) -> float:
    """Compute perceptual relative luminance from RGBA tuple."""
    r, g, b = rgba[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def load_deduped_exp1_results() -> list[dict[str, Any]]:
    """Load the 2,560 deduplicated accepted Experiment 1 dropout records (JSONL or Parquet fallback)."""
    by_id: dict[str, dict[str, Any]] = {}
    if EXP1_RESULTS_PATH.is_file():
        with EXP1_RESULTS_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                by_id[rec["call_id"]] = rec
    else:
        from common_ground.parquet_loader import load_exp1_results_from_parquet

        for rec in load_exp1_results_from_parquet(REPO_ROOT / "data/v2/hf_export"):
            by_id[rec["call_id"]] = rec
    rows = list(by_id.values())
    if len(rows) != 2560:
        raise ValueError(f"Expected 2,560 deduplicated dropout rows, got {len(rows)}")
    return rows


# =============================================================================
# FIGURE 1: Evaluation Blindness inside GPT-6.2 Astra + Do-as-I-Say Gap
# =============================================================================
def build_fig1(meta: dict[str, Any]) -> None:
    """Figure 1: Standard evaluations miss withheld constraints while peer comparisons catch them."""
    stem = "fig1_eval_blindness_within_gpt62"
    disc = meta["disclosure_meta_analysis"]
    eq1 = disc["within_gpt62_eval_blindness"]["score_eq_1_0_only"]
    shared = eq1["shared"]
    withheld = eq1["withheld"]
    contrast = eq1["contrast_shared_minus_withheld"]

    cv_diff_pp = 100.0 * (
        withheld["cross_vendor_main_7p_choice_overall"]["win_rate"]
        - shared["cross_vendor_main_7p_choice_overall"]["win_rate"]
    )
    tie_diff_pp = 100.0 * (
        withheld["cross_vendor_main_7p_choice_exact_ties"]["win_rate"]
        - shared["cross_vendor_main_7p_choice_exact_ties"]["win_rate"]
    )

    metrics = [
        {
            "key": "task_score",
            "label": "Task\nscore",
            "is_signal": False,
            "shared_pct": 100.0 * shared["mean_score"],
            "withheld_pct": 100.0 * withheld["mean_score"],
            "shared_note": f"{100.0 * shared['mean_score']:.0f}%",
            "withheld_note": f"{100.0 * withheld['mean_score']:.0f}%",
            "delta_note": None,
        },
        {
            "key": "partner_rating",
            "label": "Work-again\nrating (of 7)",
            "is_signal": False,
            "shared_pct": 100.0 * shared["cross_vendor_debriefs"]["mean_f1_again"] / 7.0,
            "withheld_pct": 100.0 * withheld["cross_vendor_debriefs"]["mean_f1_again"] / 7.0,
            "shared_note": f"{shared['cross_vendor_debriefs']['mean_f1_again']:.2f}",
            "withheld_note": f"{withheld['cross_vendor_debriefs']['mean_f1_again']:.2f}",
            "delta_note": None,
        },
        {
            "key": "keep_rate",
            "label": "Would\nkeep",
            "is_signal": False,
            "shared_pct": 100.0 * shared["cross_vendor_debriefs"]["f4_keep_rate"],
            "withheld_pct": 100.0 * withheld["cross_vendor_debriefs"]["f4_keep_rate"],
            "shared_note": f"{100.0 * shared['cross_vendor_debriefs']['f4_keep_rate']:.0f}%",
            "withheld_note": f"{100.0 * withheld['cross_vendor_debriefs']['f4_keep_rate']:.0f}%",
            "delta_note": None,
        },
        {
            "key": "cross_vendor_choice",
            "label": "Other-provider\npreference",
            "is_signal": True,
            "shared_pct": 100.0 * shared["cross_vendor_main_7p_choice_overall"]["win_rate"],
            "withheld_pct": 100.0 * withheld["cross_vendor_main_7p_choice_overall"]["win_rate"],
            "shared_note": f"{100.0 * shared['cross_vendor_main_7p_choice_overall']['win_rate']:.1f}%",
            "withheld_note": f"{100.0 * withheld['cross_vendor_main_7p_choice_overall']['win_rate']:.1f}%",
            "delta_note": f"{cv_diff_pp:+.1f} pp",
        },
        {
            "key": "exact_tie_choice",
            "label": "Exact-tie\npreference",
            "is_signal": True,
            "shared_pct": 100.0 * shared["cross_vendor_main_7p_choice_exact_ties"]["win_rate"],
            "withheld_pct": 100.0 * withheld["cross_vendor_main_7p_choice_exact_ties"]["win_rate"],
            "shared_note": f"{100.0 * shared['cross_vendor_main_7p_choice_exact_ties']['win_rate']:.1f}%",
            "withheld_note": f"{100.0 * withheld['cross_vendor_main_7p_choice_exact_ties']['win_rate']:.1f}%",
            "delta_note": f"{tie_diff_pp:+.1f} pp",
        },
    ]

    hyp = disc["producer_consumer_hypocrisy"]["all_non_aa_exact_ties"]["by_chooser"]
    core_order = ["neon4", "opus55", "astra62", "kimi31"]

    fig = plt.figure(figsize=(12.2, 4.7), facecolor=BG_WHITE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.05], left=0.072, right=0.965, top=0.74, bottom=0.14, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel A: Strategic grey for flat standard checks, Teal vs. Vermilion ONLY for peer preference
    style_axes(ax_a, grid_axis="y")
    x = np.arange(len(metrics))
    bar_w = 0.36

    for idx, m in enumerate(metrics):
        col_s = TEAL if m["is_signal"] else MUTED_LIGHT
        col_w = VERMILION if m["is_signal"] else GREY_BAR_DARK
        b_s = ax_a.bar(
            x[idx] - bar_w / 2,
            m["shared_pct"],
            width=bar_w,
            color=col_s,
            edgecolor=BG_WHITE,
            linewidth=0.7,
            zorder=3,
        )[0]
        b_w = ax_a.bar(
            x[idx] + bar_w / 2,
            m["withheld_pct"],
            width=bar_w,
            color=col_w,
            edgecolor=BG_WHITE,
            linewidth=0.7,
            zorder=3,
        )[0]

        hs = b_s.get_height()
        hw = b_w.get_height()
        ax_a.text(
            b_s.get_x() + b_s.get_width() / 2,
            hs + 2.2,
            m["shared_note"],
            ha="center",
            va="bottom",
            fontsize=8.1,
            fontweight="bold" if m["is_signal"] else "normal",
            color=TEAL_DEEP if m["is_signal"] else MUTED_DARK,
            bbox={"facecolor": BG_WHITE, "edgecolor": "none", "pad": 0.4},
            zorder=5,
        )
        ax_a.text(
            b_w.get_x() + b_w.get_width() / 2,
            hw + 2.2,
            m["withheld_note"],
            ha="center",
            va="bottom",
            fontsize=8.1,
            fontweight="bold" if m["is_signal"] else "normal",
            color=VERMILION_DEEP if m["is_signal"] else MUTED_DARK,
            bbox={"facecolor": BG_WHITE, "edgecolor": "none", "pad": 0.4},
            zorder=5,
        )
        if m["delta_note"]:
            top_y = max(hs, hw) + 12.5
            x0 = b_s.get_x() + b_s.get_width() / 2
            x1 = b_w.get_x() + b_w.get_width() / 2
            ax_a.plot([x0, x0, x1, x1], [top_y - 2.5, top_y, top_y, top_y - 2.5], color=MUTED, lw=0.9)
            ax_a.text(
                (x0 + x1) / 2,
                top_y + 1.8,
                m["delta_note"],
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
                color=VERMILION_DEEP,
            )

    ax_a.axvline(2.5, color=BORDER, linestyle=":", linewidth=1.1, zorder=1)
    ax_a.text(1.0, 117.0, "Standard checks", ha="center", va="top", fontsize=8.1, color=MUTED)
    ax_a.text(3.5, 117.0, "Anonymised peer choice", ha="center", va="top", fontsize=8.1, color=TEAL_DEEP, fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([m["label"] for m in metrics], fontsize=8.0)
    ax_a.set_ylim(0, 122)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_a.set_ylabel("Rate or % of max score")
    ax_a.set_title(
        "A. Inside GPT-6.2 Astra (perfect 1.00-score runs)",
        loc="left",
        fontsize=10.1,
        fontweight="bold",
        pad=22,
    )
    # Inline colour-coded key replacing the redundant boxed legend
    ax_a.text(0.00, 1.03, "Shared rules", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=8.2, fontweight="bold", color=TEAL_DEEP)
    ax_a.text(0.18, 1.03, "vs.", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=8.2, color=MUTED)
    ax_a.text(0.23, 1.03, "under-informed", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=8.2, fontweight="bold", color=VERMILION_DEEP)
    ax_a.text(
        0.44,
        1.03,
        f"(matched length: {shared['mean_episode_word_count']:.0f} vs. {withheld['mean_episode_word_count']:.0f} words/turn)",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.9,
        color=MUTED,
    )

    # Panel B: Dumbbell chart with zero boxed legend, unified marker palette and vermilion gap accent
    style_axes(ax_b, grid_axis="x")
    y_pos = np.arange(len(core_order))

    for i, m in enumerate(core_order):
        rec = hyp[m]
        prod = 100.0 * rec["producer_role_a_full_disclosure_rate"]
        cons = 100.0 * rec["win_rate"]
        gap = rec["hypocrisy_gap_pp_consumer_minus_producer_role_a"]
        big_gap = gap >= 50.0
        line_col = VERMILION if big_gap else MUTED_LIGHT

        if abs(cons - prod) > 0.5:
            ax_b.plot([prod, cons], [i, i], color=line_col, linewidth=2.8 if big_gap else 2.0, zorder=2)
            mid_x = (prod + cons) / 2
            ax_b.text(
                mid_x,
                i - 0.16,
                f"+{gap:.0f} pp gap",
                ha="center",
                va="bottom",
                fontsize=8.1,
                fontweight="bold" if big_gap else "normal",
                color=VERMILION_DEEP if big_gap else MUTED_DARK,
            )
        else:
            ax_b.text(
                prod,
                i - 0.18,
                "0 pp gap",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color=MUTED_DARK,
            )

        ax_b.scatter(
            prod,
            i,
            s=82 if abs(cons - prod) < 0.5 else 72,
            facecolor=BG_WHITE,
            edgecolor=TEXT_DARK,
            linewidth=2.0,
            zorder=4,
        )
        ax_b.scatter(
            cons,
            i,
            s=46 if abs(cons - prod) < 0.5 else 74,
            facecolor=TEAL,
            edgecolor=BG_WHITE,
            linewidth=0.8,
            zorder=5,
        )

        if abs(cons - prod) < 0.5:
            ax_b.text(prod, i + 0.24, f"{prod:.0f}%", ha="center", va="top", fontsize=8.0, color=TEAL_DEEP, fontweight="bold")
        else:
            ax_b.text(
                prod,
                i + 0.24,
                f"{prod:.0f}%",
                ha="center" if prod > 8 else "left",
                va="top",
                fontsize=8.0,
                color=TEXT_BODY,
                fontweight="bold",
            )
            ax_b.text(
                cons,
                i + 0.24,
                f"{cons:.1f}%",
                ha="center" if cons < 95 else "right",
                va="top",
                fontsize=8.0,
                color=TEAL_DEEP,
                fontweight="bold",
            )

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([MODEL_FULL[m] for m in core_order], fontsize=8.6, fontweight="bold", color=TEXT_DARK)
    ax_b.invert_yaxis()
    ax_b.set_xlim(-6, 108)
    ax_b.set_ylim(len(core_order) - 0.22, -0.70)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_b.set_xlabel("Rate (%)")
    ax_b.set_title(
        "B. Models under-inform yet demand partner rules",
        loc="left",
        fontsize=10.1,
        fontweight="bold",
        pad=22,
    )
    # Clean borderless inline key directly under Panel B title
    ax_b.scatter([0.02], [1.055], transform=ax_b.transAxes, clip_on=False, s=38, facecolor=BG_WHITE, edgecolor=TEXT_DARK, linewidth=1.6)
    ax_b.text(0.055, 1.03, "Shares own rules (speaking first)", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=7.8, color=TEXT_BODY)
    ax_b.scatter([0.555], [1.055], transform=ax_b.transAxes, clip_on=False, s=40, facecolor=TEAL, edgecolor=BG_WHITE, linewidth=0.6)
    ax_b.text(0.59, 1.03, "Prefers discloser (exact ties)", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=7.8, fontweight="bold", color=TEAL_DEEP)

    add_header(
        fig,
        title="Standard evaluations miss under-informing while peer comparisons catch it",
        subtitle=(
            "Config Repair: perfect 1.00-score runs by GPT-6.2 Astra (left) and what four models share vs. demand on exact ties (right)"
        ),
        left_x=0.045,
        title_fontsize=12.8,
    )

    payload = {
        "figure": stem,
        "sources": [str(META_PATH.relative_to(REPO_ROOT))],
        "panel_a_within_gpt62_score_eq_1": {
            "shared_n_episodes": shared["n_episodes"],
            "withheld_n_episodes": withheld["n_episodes"],
            "shared_mean_episode_words": shared["mean_episode_word_count"],
            "withheld_mean_episode_words": withheld["mean_episode_word_count"],
            "metrics": [
                {
                    "metric": m["key"],
                    "shared_pct": round(m["shared_pct"], 2),
                    "withheld_pct": round(m["withheld_pct"], 2),
                    "delta_note": m["delta_note"],
                }
                for m in metrics
            ],
            "contrasts": contrast,
        },
        "panel_b_asymmetry": {
            "models": [
                {
                    "model": m,
                    "display": MODEL_FULL[m],
                    "speaking_first_r1_ge5_pct": round(100.0 * hyp[m]["producer_role_a_full_disclosure_rate"], 2),
                    "exact_tie_prefers_discloser_pct": round(100.0 * hyp[m]["win_rate"], 2),
                    "wins": hyp[m]["wins"],
                    "n_discordant_ties": hyp[m]["n_discordant_ties"],
                    "gap_pp": hyp[m]["hypocrisy_gap_pp_consumer_minus_producer_role_a"],
                    "case_perm_p": hyp[m]["case_perm_p"],
                }
                for m in core_order
            ],
        },
    }
    save_figure_and_data(fig, stem, payload)


# =============================================================================
# FIGURE 2: Clarity Slopegraph (Strategic Grey + Single Vermilion Focus)
# =============================================================================
def build_fig2_clarity_slopegraph(meta: dict[str, Any]) -> None:
    """Figure 2: Automated clarity models largely reward length (Qwen4 Max falls #2 -> #7)."""
    stem = "fig2_clarity_length_bias"
    cv = meta["construct_validity"]
    r2_clarity = cv["ols_length_models"]["clarity"]["r_squared"]
    per_m = cv["per_model"]

    models_all = list(per_m.keys())
    raw_sorted = sorted(models_all, key=lambda m: per_m[m]["clarity_raw_rank"])
    adj_sorted = sorted(models_all, key=lambda m: per_m[m]["clarity_length_adjusted_rank"])

    fig = plt.figure(figsize=(9.8, 4.7), facecolor=BG_WHITE)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.76])
    ax.set_facecolor(BG_WHITE)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    x_left, x_right = 0.40, 0.62
    ax.plot([x_left, x_left], [0.85, 7.15], color=BORDER, linewidth=1.1, zorder=1)
    ax.plot([x_right, x_right], [0.85, 7.15], color=BORDER, linewidth=1.1, zorder=1)
    ax.text(
        x_left - 0.015,
        0.38,
        "Raw clarity rank\n(mean 0 to 4 score, words/turn)",
        ha="right",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT_BODY,
    )
    ax.text(
        x_right + 0.015,
        0.38,
        "Length-adjusted rank\n(residual after length, task & role)",
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT_BODY,
    )

    slope_export = []
    for m in raw_sorted:
        rec = per_m[m]
        r_raw = rec["clarity_raw_rank"]
        r_adj = rec["clarity_length_adjusted_rank"]
        is_qwen = m == "qwen4"
        col = VERMILION if is_qwen else GREY_BAR_DARK
        lw = 3.2 if is_qwen else 1.5
        alpha = 1.0 if is_qwen else 0.78

        ax.plot([x_left, x_right], [r_raw, r_adj], color=col, linewidth=lw, alpha=alpha, zorder=5 if is_qwen else 3)
        ax.scatter(
            [x_left, x_right],
            [r_raw, r_adj],
            s=56 if is_qwen else 34,
            color=col,
            edgecolor=BG_WHITE,
            linewidth=0.8,
            zorder=6 if is_qwen else 4,
        )

        # Keep 3dp on raw clarity so Kimi (2.392) vs GPT-6.2 Astra (2.387) never looks like a tie-break bug
        raw_str = fmt_half_up_3dp(rec["clarity_raw"])
        wc_int = round(rec["mean_word_count"])
        ax.text(
            x_left - 0.02,
            r_raw,
            f"#{r_raw}  {MODEL_FULL[m]}  ({raw_str}, {wc_int} words)",
            ha="right",
            va="center",
            fontsize=8.5,
            fontweight="bold" if is_qwen else "normal",
            color=VERMILION_DEEP if is_qwen else TEXT_BODY,
        )
        ax.text(
            x_right + 0.02,
            r_adj,
            f"#{r_adj}  {MODEL_FULL[m]}  ({rec['clarity_resid']:+.3f})",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold" if is_qwen else "normal",
            color=VERMILION_DEEP if is_qwen else TEXT_BODY,
        )
        slope_export.append(
            {
                "model": m,
                "display": MODEL_FULL[m],
                "mean_word_count": rec["mean_word_count"],
                "clarity_raw": rec["clarity_raw"],
                "clarity_raw_rank": r_raw,
                "clarity_resid": rec["clarity_resid"],
                "clarity_length_adjusted": rec["clarity_length_adjusted"],
                "clarity_length_adjusted_rank": r_adj,
            }
        )

    # Borderless inline slope annotation in open white pocket beside Qwen4 Max's plunge
    ax.text(
        0.528,
        4.48,
        "-5 ranks",
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=VERMILION_DEEP,
        bbox={"facecolor": BG_WHITE, "edgecolor": "none", "pad": 1.5},
        zorder=7,
    )

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(7.55, -0.35)

    add_header(
        fig,
        title="Automated clarity models largely reward length: verbose Qwen4 Max falls from #2 to #7",
        subtitle=(
            f"Word count, task and role explain {100.0 * r2_clarity:.1f}% of clarity variance "
            f"across {cv['n_turns']:,} live turns scored by three independent decision models"
        ),
        left_x=0.04,
        title_fontsize=12.4,
    )

    payload = {
        "figure": stem,
        "sources": [str(META_PATH.relative_to(REPO_ROOT))],
        "n_turns": cv["n_turns"],
        "ols_r_squared": r2_clarity,
        "models": slope_export,
        "raw_rank_order": raw_sorted,
        "adjusted_rank_order": adj_sorted,
    }
    save_figure_and_data(fig, stem, payload)


# =============================================================================
# FIGURE 3: Causal Double Dissociation (Pruned 5-Row Forest + Matching Dot-Whisker)
# =============================================================================
def build_fig3_causal_dissociation(exp2: dict[str, Any]) -> None:
    """Figure 3: Adding omitted private rules shifts peer preference while rewriting tone has zero effect."""
    stem = "fig3_causal_double_dissociation"
    cap_cv = exp2["capstone_comparison"]["itt_cross_vendor"]
    plac_cv = exp2["placebo_arm"]["itt_cross_vendor"]["overall"]
    main_cv = exp2["main_arm"]["itt_cross_vendor"]["overall"]
    gp_cv = exp2["main_arm"]["itt_cross_vendor"]["by_task"]["goalpost"]
    tr_cv = exp2["main_arm"]["itt_cross_vendor"]["by_task"]["triage"]

    rows = [
        {
            "id": "tone_rewrite_cv",
            "label": f"Tone rewrite: warm vs. blunt (n={cap_cv['n_cases']})",
            "delta_pp": 100.0 * cap_cv["mean_delta_warm"],
            "ci_lo_pp": 100.0 * cap_cv["delta_warm_pair_boot_ci95"][0],
            "ci_hi_pp": 100.0 * cap_cv["delta_warm_pair_boot_ci95"][1],
            "p": cap_cv["pair_sign_flip_perm_p"],
            "toward": cap_cv["n_switched_toward_warm"],
            "away": cap_cv["n_switched_away_from_warm"],
            "n_cases": cap_cv["n_cases"],
            "color": GREY_BAR_DARK,
            "bold": False,
        },
        {
            "id": "placebo_cv",
            "label": f"Length-matched public recap, placebo (n={plac_cv['n_cases']})",
            "delta_pp": 100.0 * plac_cv["mean_delta_echo"],
            "ci_lo_pp": 100.0 * plac_cv["delta_echo_pair_boot_ci95"][0],
            "ci_hi_pp": 100.0 * plac_cv["delta_echo_pair_boot_ci95"][1],
            "p": plac_cv["pair_sign_flip_perm_p"],
            "toward": plac_cv["n_switched_toward_echo"],
            "away": plac_cv["n_switched_away_from_echo"],
            "n_cases": plac_cv["n_cases"],
            "color": GREY_BAR_DARK,
            "bold": False,
        },
        {
            "id": "state_injection_cv",
            "label": f"Added omitted private rules: overall (n={main_cv['n_cases']})",
            "delta_pp": 100.0 * main_cv["mean_delta_info"],
            "ci_lo_pp": 100.0 * main_cv["delta_info_pair_boot_ci95"][0],
            "ci_hi_pp": 100.0 * main_cv["delta_info_pair_boot_ci95"][1],
            "p": main_cv["pair_sign_flip_perm_p"],
            "toward": main_cv["n_switched_toward_info"],
            "away": main_cv["n_switched_away_from_info"],
            "n_cases": main_cv["n_cases"],
            "color": TEAL,
            "bold": True,
        },
        {
            "id": "state_config_repair_cv",
            "label": f"Config Repair: Round-1 handoff (n={gp_cv['n_cases']})",
            "delta_pp": 100.0 * gp_cv["mean_delta_info"],
            "ci_lo_pp": 100.0 * gp_cv["delta_info_pair_boot_ci95"][0],
            "ci_hi_pp": 100.0 * gp_cv["delta_info_pair_boot_ci95"][1],
            "p": gp_cv["pair_sign_flip_perm_p"],
            "toward": gp_cv["n_switched_toward_info"],
            "away": gp_cv["n_switched_away_from_info"],
            "n_cases": gp_cv["n_cases"],
            "color": TEAL_SOFT,
            "bold": False,
            "subrow": True,
        },
        {
            "id": "state_triage_cv",
            "label": f"Incident Triage: full transcript (n={tr_cv['n_cases']})",
            "delta_pp": 100.0 * tr_cv["mean_delta_info"],
            "ci_lo_pp": 100.0 * tr_cv["delta_info_pair_boot_ci95"][0],
            "ci_hi_pp": 100.0 * tr_cv["delta_info_pair_boot_ci95"][1],
            "p": tr_cv["pair_sign_flip_perm_p"],
            "toward": tr_cv["n_switched_toward_info"],
            "away": tr_cv["n_switched_away_from_info"],
            "n_cases": tr_cv["n_cases"],
            "color": TEAL_SOFT,
            "bold": False,
            "subrow": True,
        },
    ]

    rev_dict = exp2["main_arm"]["itt_all_reviewers"]["by_reviewer"]
    rev_order = sorted(rev_dict.keys(), key=lambda r: rev_dict[r]["mean_delta_info"], reverse=True)

    fig = plt.figure(figsize=(12.4, 4.7), facecolor=BG_WHITE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.05], left=0.285, right=0.965, top=0.745, bottom=0.145, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel A: Clean 5-row forest plot (other-provider evaluators), axis to +75 so +61.5 CI never clips
    style_axes(ax_a, grid_axis="x")
    y_a = np.arange(len(rows))
    ax_a.axvline(0.0, color=TEXT_BODY, linestyle="--", linewidth=1.0, zorder=1)
    ax_a.axvline(72.0, color=BORDER, linestyle="-", linewidth=0.8, zorder=1)
    ax_a.axhspan(1.5, 2.5, color=ROW_TINT, zorder=0)
    ax_a.axhline(1.5, color=MUTED_LIGHT, linestyle="-", linewidth=0.9, zorder=1)

    for i, r in enumerate(rows):
        d = r["delta_pp"]
        lo = r["ci_lo_pp"]
        hi = r["ci_hi_pp"]
        ax_a.hlines(i, lo, hi, color=r["color"], linewidth=2.8 if r["bold"] else 2.1, zorder=3)
        ax_a.plot([lo, lo], [i - 0.13, i + 0.13], color=r["color"], linewidth=1.2, zorder=3)
        ax_a.plot([hi, hi], [i - 0.13, i + 0.13], color=r["color"], linewidth=1.2, zorder=3)
        ax_a.scatter(
            d,
            i,
            s=80 if r["bold"] else 58,
            color=r["color"],
            edgecolor=BG_WHITE,
            linewidth=0.9,
            zorder=4,
        )

        ax_a.text(
            91.0,
            i,
            f"{d:+.1f} pp",
            ha="right",
            va="center",
            fontsize=8.2,
            fontweight="bold" if r["bold"] else "normal",
            color=TEAL_DEEP if r["bold"] else TEXT_BODY,
        )
        ax_a.text(
            120.0,
            i,
            f"{r['toward']} / {r['away']}",
            ha="right",
            va="center",
            fontsize=8.1,
            fontweight="bold" if r["bold"] else "normal",
            color=TEXT_DARK if r["toward"] > 0 else MUTED,
        )

    ax_a.text(91.0, -0.64, "Shift", ha="right", va="bottom", fontsize=7.8, fontweight="bold", color=MUTED)
    ax_a.text(120.0, -0.64, "For / against", ha="right", va="bottom", fontsize=7.8, fontweight="bold", color=MUTED)

    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels([r["label"] for r in rows], fontsize=8.3)
    for tick_lbl, r in zip(ax_a.get_yticklabels(), rows, strict=True):
        if r["bold"]:
            tick_lbl.set_fontweight("bold")
            tick_lbl.set_color(TEXT_DARK)
        elif r.get("subrow"):
            tick_lbl.set_color(MUTED_DARK)
            tick_lbl.set_fontsize(8.0)
    ax_a.invert_yaxis()
    ax_a.set_xlim(-14, 124)
    ax_a.set_ylim(len(rows) - 0.35, -0.85)
    ax_a.spines["bottom"].set_bounds(-14, 72)
    ax_a.set_xticks([0, 20, 40, 60])
    ax_a.set_xticklabels(["0", "+20", "+40", "+60"])
    ax_a.set_xlabel("Preference shift towards edited candidate (pp, 95% CI)", x=0.31)
    ax_a.set_title(
        "A. Tone rewrite vs. public recap vs. omitted private rules",
        loc="left",
        fontsize=9.9,
        fontweight="bold",
        pad=24,
    )
    ax_a.text(0.0, 1.045, "Other-provider evaluators (all-evaluator numbers in Table C1)", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=7.9, color=MUTED)

    # Panel B: Unified dot-and-whisker grammar matching Panel A, extended to +80 so +70.8 CI never overflows
    style_axes(ax_b, grid_axis="x")
    y_b = np.arange(len(rev_order))
    ax_b.axvline(0.0, color=TEXT_BODY, linestyle="--", linewidth=1.0, zorder=1)

    for i, rev in enumerate(rev_order):
        rec = rev_dict[rev]
        d = 100.0 * rec["mean_delta_info"]
        lo = 100.0 * rec["delta_info_pair_boot_ci95"][0]
        hi = 100.0 * rec["delta_info_pair_boot_ci95"][1]
        p_val = rec["pair_sign_flip_perm_p"]
        sig = p_val < 0.05
        col = TEAL if sig else TEAL_SOFT

        ax_b.hlines(i, lo, hi, color=col, linewidth=2.6 if sig else 2.0, zorder=3)
        ax_b.plot([lo, lo], [i - 0.12, i + 0.12], color=col, linewidth=1.2, zorder=3)
        ax_b.plot([hi, hi], [i - 0.12, i + 0.12], color=col, linewidth=1.2, zorder=3)
        ax_b.scatter(
            d,
            i,
            s=76 if sig else 58,
            facecolor=col,
            edgecolor=BG_WHITE,
            linewidth=0.9,
            zorder=4,
        )
        ax_b.text(
            hi + 2.8,
            i,
            f"{d:+.1f} pp",
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold" if sig else "normal",
            color=TEAL_DEEP if sig else TEXT_BODY,
        )

    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels([MODEL_FULL[r] for r in rev_order], fontsize=8.4, fontweight="bold", color=TEXT_DARK)
    ax_b.invert_yaxis()
    ax_b.set_xlim(-12, 96)
    ax_b.set_ylim(len(rev_order) - 0.35, -0.60)
    ax_b.set_xticks([0, 20, 40, 60, 80])
    ax_b.set_xticklabels(["0", "+20", "+40", "+60", "+80"])
    ax_b.set_xlabel("Preference shift towards private rules (pp, 95% CI)")
    ax_b.set_title(
        "B. Shift by evaluator (private rules)",
        loc="left",
        fontsize=9.9,
        fontweight="bold",
        pad=24,
    )
    ax_b.text(0.0, 1.045, "All 4 models favour private rules (24 pairs each)", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=7.9, color=MUTED)

    add_header(
        fig,
        title="Adding omitted private rules shifts peer preference while tone and recaps do not",
        subtitle="Blind within-pair crossover trials: mean preference shift (95% CI) and order-consistent flips (for / against)",
        left_x=0.04,
        title_fontsize=12.6,
    )

    payload = {
        "figure": stem,
        "sources": [str(EXP2_PATH.relative_to(REPO_ROOT))],
        "panel_a_forest_rows": rows,
        "panel_b_per_evaluator_pooled": [
            {
                "evaluator": r,
                "display": MODEL_FULL[r],
                "n_cases": rev_dict[r]["n_cases"],
                "n_pairs": rev_dict[r]["n_pairs"],
                "mean_delta_info_pp": round(100.0 * rev_dict[r]["mean_delta_info"], 2),
                "ci95_pp": [round(100.0 * x, 2) for x in rev_dict[r]["delta_info_pair_boot_ci95"]],
                "pair_sign_flip_perm_p": rev_dict[r]["pair_sign_flip_perm_p"],
                "n_switched_toward_info": rev_dict[r]["n_switched_toward_info"],
                "n_switched_away_from_info": rev_dict[r]["n_switched_away_from_info"],
            }
            for r in rev_order
        ],
    }
    save_figure_and_data(fig, stem, payload)


# =============================================================================
# FIGURE 4: Partner Drops Out (Cleaned Sender Survival + Strictly Sorted Ladder)
# =============================================================================
def build_fig4_dropout(exp1: dict[str, Any], gate: dict[str, Any], exp1_rows: list[dict[str, Any]]) -> None:
    """Figure 4: When a lead drops out, upstream rule sharing governs survival across every model tier."""
    stem = "fig4_dropout_sender_and_ladder"
    by_sender = exp1["by_sender"]
    senders_sorted = sorted(
        by_sender.keys(),
        key=lambda s: (
            by_sender[s]["primary_role_a_cross_vendor"]["strict_all_pass_rate"],
            by_sender[s]["primary_role_a_cross_vendor"]["mean_sender_r1_r2_n_formula"],
        ),
        reverse=True,
    )

    wa_r1 = exp1["within_astra62_shared_vs_withheld"]["primary_role_a_cross_vendor_r1"]
    wa = exp1["within_astra62_shared_vs_withheld"]["primary_role_a_cross_vendor_r1_r2"]
    s_n = wa["shared"]["n_cases"]
    s_pass = round(wa["shared"]["strict_all_pass_rate"] * s_n)
    w_n = wa["withheld"]["n_cases"]
    w_pass = round(wa["withheld"]["strict_all_pass_rate"] * w_n)
    r1_s_n = wa_r1["shared"]["n_cases"]
    r1_s_pass = round(wa_r1["shared"]["strict_all_pass_rate"] * r1_s_n)
    r1_w_n = wa_r1["withheld"]["n_cases"]
    r1_w_pass = round(wa_r1["withheld"]["strict_all_pass_rate"] * r1_w_n)

    role_a_rows = [r for r in exp1_rows if r["dropped_role"] == "A"]
    if len(role_a_rows) != 1280:
        raise ValueError(f"Expected 1,280 primary first-speaker rows, got {len(role_a_rows)}")

    roster = exp1["locked_roster"]
    # Strictly sorted by Artificial Analysis Intelligence Index v5 within each tier (62 -> 61 -> 59 -> 57, then 27 -> 19 -> 19 -> unscored)
    receiver_order = [
        "astra62",
        "neon4",
        "opus55",
        "astra60",
        "hydrogen",
        "deepseek_flash",
        "haiku55",
        "gpt6_luna",
    ]

    ladder_rows = []
    for r in receiver_order:
        g_locked = gate["receivers"][r]["locked"]
        floor_rate = g_locked["gate_floor_strict_pass_rate"]
        ceil_rate = g_locked["gate_ceiling_strict_pass_rate"]
        sub = [row for row in role_a_rows if row["receiver_key"] == r]
        sh = [row for row in sub if row["sender_r1_r2_n_formula"] >= 5]
        wh = [row for row in sub if row["sender_r1_r2_n_formula"] < 5]
        wh_ok = [row for row in wh if row["status"] == "ok"]

        sh_pct = 100.0 * sum(1 for row in sh if row["strict_all_pass"]) / len(sh)
        wh_pct = 100.0 * sum(1 for row in wh if row["strict_all_pass"]) / len(wh)
        wh_ok_pct = 100.0 * sum(1 for row in wh_ok if row["strict_all_pass"]) / len(wh_ok)
        wh_trunc = len(wh) - len(wh_ok)

        ladder_rows.append(
            {
                "receiver": r,
                "display": MODEL_FULL[r],
                "label": RECEIVER_LADDER_LABEL[r],
                "tier": roster[r]["tier"],
                "aa_index_v5": roster[r]["locked_aa_index_v5"],
                "floor_pct": 100.0 * floor_rate,
                "withheld_pct": wh_pct,
                "withheld_ok_pct": wh_ok_pct,
                "withheld_truncations": wh_trunc,
                "withheld_n": len(wh),
                "shared_pct": sh_pct,
                "shared_n": len(sh),
                "ceiling_pct": 100.0 * ceil_rate,
            }
        )

    fig = plt.figure(figsize=(13.2, 5.75), facecolor=BG_WHITE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.18], left=0.135, right=0.97, top=0.70, bottom=0.145, wspace=0.56)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel A: Clean dropped-teammate bars with explicit colour-coded group headings
    style_axes(ax_a, grid_axis="x")
    y_s = np.array([0.45, 1.45, 2.45, 3.45, 4.45, 6.20, 7.20])
    sender_data = []

    for i, s in enumerate(senders_sorted):
        yp_s = y_s[i]
        rec = by_sender[s]["primary_role_a_cross_vendor"]
        itt = 100.0 * rec["strict_all_pass_rate"]
        ci_lo = 100.0 * rec["strict_all_pass_ci95"][0]
        ci_hi = 100.0 * rec["strict_all_pass_ci95"][1]
        ok_rate = 100.0 * rec["strict_all_pass_rate_ok"]
        own_ok = 100.0 * rec["own_plus_r3_pass_rate_ok"]
        rules = rec["mean_sender_r1_r2_n_formula"]
        is_withhold = itt < 90.0
        col = VERMILION if is_withhold else TEAL

        ax_a.barh(yp_s, itt, height=0.58, color=col, alpha=0.90, edgecolor=BG_WHITE, linewidth=0.6, zorder=3)
        ax_a.hlines(yp_s, ci_lo, ci_hi, color=TEXT_DARK, linewidth=1.35, zorder=4)
        ax_a.plot([ci_lo, ci_lo], [yp_s - 0.13, yp_s + 0.13], color=TEXT_DARK, linewidth=1.0, zorder=4)
        ax_a.plot([ci_hi, ci_hi], [yp_s - 0.13, yp_s + 0.13], color=TEXT_DARK, linewidth=1.0, zorder=4)

        ax_a.text(
            104.5,
            yp_s,
            f"{itt:.1f}%",
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold" if is_withhold else "normal",
            color=VERMILION_DEEP if is_withhold else TEXT_DARK,
        )
        ax_a.text(
            182.0,
            yp_s,
            f"{rules:.1f}/7",
            ha="right",
            va="center",
            fontsize=8.1,
            color=VERMILION_DEEP if is_withhold else MUTED_DARK,
            fontweight="bold" if is_withhold else "normal",
        )

        sender_data.append(
            {
                "sender": s,
                "display": MODEL_FULL[s],
                "overall_pass_pct": round(itt, 2),
                "overall_pass_ci95_pct": [round(ci_lo, 2), round(ci_hi, 2)],
                "completed_only_pass_pct": round(ok_rate, 2),
                "own_plus_r3_ok_pct": round(own_ok, 2),
                "mean_r1_r2_rules": round(rules, 2),
                "n_cases": rec["n_cases"],
                "n_truncations": rec["n_truncation_errors"],
            }
        )

    ax_a.text(2.0, -0.34, "Usually shared rules", ha="left", va="bottom", fontsize=7.6, fontweight="bold", color=TEAL_DEEP, bbox={"facecolor": BG_WHITE, "edgecolor": "none", "pad": 1.0}, zorder=5)
    ax_a.text(104.5, -0.34, "Completion", ha="left", va="bottom", fontsize=7.7, fontweight="bold", color=MUTED)
    ax_a.text(182.0, -0.34, "Avg. rules shared", ha="right", va="bottom", fontsize=7.7, fontweight="bold", color=MUTED)
    ax_a.axhline(5.16, color=MUTED_LIGHT, linestyle="--", linewidth=0.95, zorder=1)
    ax_a.text(2.0, 5.36, "Usually under-informed", ha="left", va="top", fontsize=7.6, fontweight="bold", color=VERMILION_DEEP, bbox={"facecolor": BG_WHITE, "edgecolor": "none", "pad": 1.0}, zorder=5)

    ax_a.set_yticks(y_s)
    ax_a.set_yticklabels([MODEL_FULL[s] for s in senders_sorted], fontsize=8.4, fontweight="bold", color=TEXT_DARK)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 186)
    ax_a.set_ylim(7.80, -0.82)
    ax_a.spines["bottom"].set_bounds(0, 100)
    ax_a.set_xticks([0, 20, 40, 60, 80, 100])
    ax_a.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax_a.set_xlabel("Task completion rate (%, 95% CI)", x=0.27)
    ax_a.set_title(
        "A. Task completion by dropped teammate",
        loc="left",
        fontsize=10.0,
        fontweight="bold",
        pad=32,
    )
    ax_a.text(0.0, 1.03, "Other-provider takeover models (n = 1,036, solved 99% of own rules)", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=7.8, color=MUTED)

    # Panel B: Strictly sorted capability ladder with crystal-clear 2-row inline key
    style_axes(ax_b, grid_axis="x")
    y_b = [0.25, 1.25, 2.25, 3.25, 5.55, 6.55, 7.55, 8.55]

    ax_b.axhline(4.0, color=MUTED_LIGHT, linestyle="--", linewidth=0.9, zorder=1)
    ax_b.text(1.5, -0.78, "FRONTIER TAKEOVER MODELS (medium reasoning, Index 57 to 62)", ha="left", va="center", fontsize=7.2, fontweight="bold", color=MUTED, bbox={"facecolor": BG_WHITE, "edgecolor": "none", "pad": 1.0}, zorder=5)
    ax_b.text(1.5, 4.38, "LOW-COST WORKER MODELS (low / off reasoning, Index 19 to 27 + Luna)", ha="left", va="center", fontsize=7.2, fontweight="bold", color=MUTED, bbox={"facecolor": BG_WHITE, "edgecolor": "none", "pad": 1.0}, zorder=5)

    for yp, row in zip(y_b, ladder_rows, strict=True):
        fl = row["floor_pct"]
        wh = row["withheld_pct"]
        sh = row["shared_pct"]
        ce = row["ceiling_pct"]

        ax_b.plot([fl, ce], [yp, yp], color=BORDER, linewidth=1.8, zorder=2)
        ax_b.plot([wh, sh], [yp, yp], color=TEAL_SOFT, linewidth=2.6, alpha=0.60, zorder=2)

        ax_b.scatter(fl, yp, marker="s", s=34, color=MUTED_LIGHT, edgecolor=BG_WHITE, linewidth=0.5, zorder=3)
        ax_b.scatter(ce, yp, marker="D", s=84, facecolor="none", edgecolor=TEXT_BODY, linewidth=1.35, zorder=4)
        ax_b.scatter(wh, yp, marker="o", s=58, color=VERMILION, edgecolor=BG_WHITE, linewidth=0.6, zorder=5)
        ax_b.scatter(sh, yp, marker="o", s=56, color=TEAL, edgecolor=BG_WHITE, linewidth=0.6, zorder=6)

        if row["receiver"] == "neon4":
            ax_b.scatter(row["withheld_ok_pct"], yp, marker="o", s=40, facecolor=BG_WHITE, edgecolor=VERMILION, linewidth=1.4, zorder=5)
            ax_b.text(wh, yp - 0.27, f"{wh:.0f}%*", ha="center", va="bottom", fontsize=7.5, color=VERMILION_DEEP, fontweight="bold")
            ax_b.text(row["withheld_ok_pct"] - 1.2, yp - 0.27, f"{row['withheld_ok_pct']:.0f}% finished", ha="right", va="bottom", fontsize=6.9, color=MUTED_DARK)
        else:
            ax_b.text(wh, yp - 0.27, f"{wh:.0f}%", ha="center", va="bottom", fontsize=7.5, color=VERMILION_DEEP, fontweight="bold")

        sh_lbl = "100%" if sh >= 99.95 else f"{sh:.1f}%"
        ax_b.text(
            sh - 1.2 if sh > 97 else sh,
            yp - 0.27,
            sh_lbl,
            ha="right" if sh > 97 else "center",
            va="bottom",
            fontsize=7.4,
            color=TEAL_DEEP,
            fontweight="bold",
        )

    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels([r["label"] for r in ladder_rows], fontsize=7.9, color=TEXT_DARK)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 106)
    ax_b.set_ylim(9.20, -1.20)
    ax_b.set_xticks([0, 20, 40, 60, 80, 100])
    ax_b.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax_b.set_xlabel("Task completion rate on first-speaker handoffs (%)")
    ax_b.set_title(
        "B. Task completion by model taking over",
        loc="left",
        fontsize=10.0,
        fontweight="bold",
        pad=32,
    )
    # Clean 2-row borderless inline key above Panel B explaining every symbol plainly
    ax_b.scatter([0.015], [1.095], transform=ax_b.transAxes, clip_on=False, marker="s", s=28, color=MUTED_LIGHT)
    ax_b.text(0.042, 1.073, "No transcript (baseline)", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=7.4, color=MUTED_DARK)
    ax_b.scatter([0.465], [1.095], transform=ax_b.transAxes, clip_on=False, marker="D", s=34, facecolor="none", edgecolor=TEXT_BODY, linewidth=1.2)
    ax_b.text(0.492, 1.073, "Verbatim rules (upper bound)", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=7.4, color=TEXT_BODY)

    ax_b.scatter([0.015], [1.036], transform=ax_b.transAxes, clip_on=False, marker="o", s=34, color=VERMILION)
    ax_b.text(0.042, 1.014, "Teammate under-informed (<5, n=44)", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=7.4, fontweight="bold", color=VERMILION_DEEP)
    ax_b.scatter([0.565], [1.036], transform=ax_b.transAxes, clip_on=False, marker="o", s=34, color=TEAL)
    ax_b.text(0.592, 1.014, "Teammate shared rules (≥5, n=116)", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=7.4, fontweight="bold", color=TEAL_DEEP)

    fig.text(
        0.97,
        0.024,
        "*Gemini 4 Neon hit the 16k output token limit on 12 of 44 under-informed handoffs (open circle = 88% on finished replies)",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=MUTED_DARK,
    )

    add_header(
        fig,
        title="When a teammate drops out, proactive rule sharing separates success from failure",
        subtitle="1,280 first-speaker handoffs: other-provider completion by dropped teammate (left) and all handoffs by takeover model (right)",
        left_x=0.04,
        title_fontsize=12.6,
    )

    payload = {
        "figure": stem,
        "sources": [
            str(EXP1_PATH.relative_to(REPO_ROOT)),
            str(GATE_PATH.relative_to(REPO_ROOT)),
            "data/v2/hf_export/exp1_dropout.parquet",
        ],
        "panel_a_sender_primary_first_speaker_other_provider": sender_data,
        "within_gpt62_first_speaker_split_r1": {
            "shared_strict_pct": round(100.0 * wa_r1["shared"]["strict_all_pass_rate"], 2),
            "shared_pass": r1_s_pass,
            "shared_n": r1_s_n,
            "shared_episodes": wa_r1["n_episodes_shared"],
            "withheld_strict_pct": round(100.0 * wa_r1["withheld"]["strict_all_pass_rate"], 2),
            "withheld_pass": r1_w_pass,
            "withheld_n": r1_w_n,
            "withheld_episodes": wa_r1["n_episodes_withheld"],
            "delta_strict_pp": wa_r1["delta_strict_pp"],
            "delta_strict_ci95_pp": wa_r1["delta_strict_ci95_pp"],
            "exact_episode_perm_p": wa_r1["episode_perm_p_strict"],
        },
        "within_gpt62_first_speaker_split_r1_r2": {
            "shared_strict_pct": round(100.0 * wa["shared"]["strict_all_pass_rate"], 2),
            "shared_pass": s_pass,
            "shared_n": s_n,
            "shared_episodes": wa["n_episodes_shared"],
            "withheld_strict_pct": round(100.0 * wa["withheld"]["strict_all_pass_rate"], 2),
            "withheld_pass": w_pass,
            "withheld_n": w_n,
            "withheld_episodes": wa["n_episodes_withheld"],
            "delta_strict_pp": wa["delta_strict_pp"],
            "delta_strict_ci95_pp": wa["delta_strict_ci95_pp"],
            "exact_episode_perm_p": wa["episode_perm_p_strict"],
        },
        "panel_b_receiver_capability_ladder_actual_disclosure": ladder_rows,
    }
    save_figure_and_data(fig, stem, payload)


# =============================================================================
# FIGURE 5: 7 Senders x 8 Receivers Survival Matrix (Honest 0%-100%, Top Headers)
# =============================================================================
def build_fig5_heatmap(exp1: dict[str, Any]) -> None:
    """Figure 5: Cheap workers match frontier receivers when rules are shared, with no same-family edge."""
    stem = "fig5_sender_x_receiver_heatmap"
    mat_dict = exp1["sender_x_receiver_matrix"]["primary_role_a"]
    by_sender = exp1["by_sender"]

    senders = sorted(
        by_sender.keys(),
        key=lambda s: (
            by_sender[s]["primary_role_a_cross_vendor"]["strict_all_pass_rate"],
            by_sender[s]["primary_role_a_cross_vendor"]["mean_sender_r1_r2_n_formula"],
        ),
        reverse=True,
    )
    receivers = [
        "astra62",
        "neon4",
        "opus55",
        "astra60",
        "hydrogen",
        "deepseek_flash",
        "haiku55",
        "gpt6_luna",
    ]

    grid = np.zeros((len(senders), len(receivers)))
    cells_export: list[dict[str, Any]] = []
    for i, s in enumerate(senders):
        for j, r in enumerate(receivers):
            cell = mat_dict[s][r]
            grid[i, j] = cell["strict_all_pass_rate"]
            cells_export.append(
                {
                    "sender": s,
                    "sender_display": MODEL_FULL[s],
                    "receiver": r,
                    "receiver_display": MODEL_FULL[r],
                    "same_family": MODEL_FAMILY[s] == MODEL_FAMILY[r],
                    "n": cell["n"],
                    "n_ok": cell["n_ok"],
                    "n_truncation_errors": cell["n_truncation_errors"],
                    "strict_all_pass_rate": cell["strict_all_pass_rate"],
                    "strict_all_pass_rate_ok": cell["strict_all_pass_rate_ok"],
                }
            )

    fig = plt.figure(figsize=(12.2, 6.05), facecolor=BG_WHITE)
    ax = fig.add_axes([0.285, 0.115, 0.59, 0.545])
    cax = fig.add_axes([0.895, 0.115, 0.017, 0.545])

    # Honest 0% to 100% scale with CVD-safe calm-high / dark-alert sequential ramp
    vmin_val, vmax_val = 0.00, 1.00
    im = ax.imshow(grid, cmap=ALERT_SEQ_CMAP, vmin=vmin_val, vmax=vmax_val, aspect="auto")

    for x_edge in np.arange(-0.5, len(receivers), 1.0):
        ax.axvline(x_edge, color=BG_WHITE, linewidth=1.8, zorder=2)
    for y_edge in np.arange(-0.5, len(senders), 1.0):
        ax.axhline(y_edge, color=BG_WHITE, linewidth=1.8, zorder=2)

    ax.axvline(3.5, color=TEXT_DARK, linewidth=1.7, zorder=4)
    ax.axhline(4.5, color=TEXT_DARK, linestyle="--", linewidth=1.35, zorder=4)

    for i, s in enumerate(senders):
        for j, r in enumerate(receivers):
            cell = mat_dict[s][r]
            val = cell["strict_all_pass_rate"]
            rgba = ALERT_SEQ_CMAP(np.clip(val, vmin_val, vmax_val))
            txt_col = BG_WHITE if luminance(rgba) < 0.52 else TEXT_DARK
            n_t = cell["n_truncation_errors"]
            txt = f"{100.0 * val:.0f}%*" if n_t > 0 else f"{100.0 * val:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.8, fontweight="bold", color=txt_col, zorder=5)

            # Inset dashed border for same-family pairs (never collides with central divider)
            if MODEL_FAMILY[s] == MODEL_FAMILY[r]:
                rect = mpatches.Rectangle(
                    (j - 0.42, i - 0.40),
                    0.84,
                    0.80,
                    fill=False,
                    edgecolor=TEXT_DARK,
                    linestyle="--",
                    linewidth=1.35,
                    zorder=6,
                )
                ax.add_patch(rect)

    # Top column headers directly under tier banners (banners strictly ABOVE model names)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(receivers)))
    ax.set_xticklabels([RECEIVER_COL_LABEL[r] for r in receivers], fontsize=7.6, color=TEXT_DARK, linespacing=1.12)
    ax.tick_params(axis="x", top=False, bottom=False, pad=5)

    ax.text(1.5, 1.245, "FRONTIER TAKEOVER MODELS (medium reasoning)", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.1, fontweight="bold", color=TEXT_BODY, clip_on=False)
    ax.text(5.5, 1.245, "LOW-COST WORKER MODELS (low / off reasoning)", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.1, fontweight="bold", color=TEXT_BODY, clip_on=False)
    ax.plot([-0.42, 3.38], [1.215, 1.215], transform=ax.get_xaxis_transform(), color=BORDER, linewidth=1.0, clip_on=False)
    ax.plot([3.62, 7.42], [1.215, 1.215], transform=ax.get_xaxis_transform(), color=BORDER, linewidth=1.0, clip_on=False)

    sender_ylabels = [
        f"{MODEL_FULL[s]}  (n={mat_dict[s][receivers[0]]['n']})" for s in senders
    ]
    ax.set_yticks(np.arange(len(senders)))
    ax.set_yticklabels(sender_ylabels, fontsize=8.3, fontweight="bold", color=TEXT_DARK)
    ax.tick_params(axis="y", left=False, pad=6)

    # Clear left-margin group brackets aligned cleanly inside left_x=0.04
    ax.text(-0.56, -0.64, "Dropped teammate (Player A)", ha="right", va="bottom", fontsize=8.2, fontweight="bold", color=MUTED, clip_on=False)
    ax.plot([-0.32, -0.32], [0.31, 0.96], transform=ax.transAxes, color=TEAL, linewidth=2.2, clip_on=False)
    ax.text(-0.34, 0.635, "Usually\nshared\nrules", transform=ax.transAxes, ha="right", va="center", fontsize=7.5, fontweight="bold", color=TEAL_DEEP, clip_on=False)
    ax.plot([-0.32, -0.32], [0.03, 0.255], transform=ax.transAxes, color=VERMILION, linewidth=2.2, clip_on=False)
    ax.text(-0.34, 0.142, "Usually\nunder-\ninformed", transform=ax.transAxes, ha="right", va="center", fontsize=7.5, fontweight="bold", color=VERMILION_DEEP, clip_on=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
    cb.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    cb.ax.tick_params(labelsize=8.0, colors=TEXT_TICK, length=2.5)
    cb.outline.set_edgecolor(BORDER)
    cb.outline.set_linewidth(0.8)
    cb.set_label("Task completion (%)", fontsize=8.4, color=TEXT_BODY, labelpad=6)

    legend_box = mpatches.Patch(facecolor="none", edgecolor=TEXT_DARK, linestyle="--", linewidth=1.35, label="Same model family (no decoding edge)")
    ax.legend(
        handles=[legend_box],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.042),
        frameon=False,
        fontsize=7.9,
        handlelength=1.5,
        handleheight=1.05,
        borderaxespad=0.0,
    )
    ax.text(
        1.0,
        -0.056,
        "*Gemini 4 Neon hit 16k output token limit (91% / 90% on finished replies)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color=MUTED_DARK,
        clip_on=False,
    )

    add_header(
        fig,
        title="Low-cost workers match frontier takeover models when rules are shared",
        subtitle="1,280 first-speaker handoffs across all 7 dropped teammates (rows) and 8 takeover models (columns)",
        left_x=0.04,
        title_fontsize=12.6,
    )

    payload = {
        "figure": stem,
        "sources": [str(EXP1_PATH.relative_to(REPO_ROOT))],
        "senders_order": senders,
        "receivers_order": receivers,
        "cells": cells_export,
    }
    save_figure_and_data(fig, stem, payload)


# =============================================================================
# FIGURE 6: 4-Column Tufte Small Multiples Compute Bloat Tax
# =============================================================================
def build_fig6_bloat_tax(meta: dict[str, Any]) -> None:
    """Figure 6: Four-column small multiples showing partner-induced compute and API cost overhead."""
    stem = "fig6_compute_bloat_tax"
    bt = meta["bloat_tax_and_contagion"]["exposed_turns_primary"]["overall"]["pct_delta_by_metric"]
    cost_by_p = bt["cost_usd"]["by_partner"]
    own_words_by_p = {k: v["mean_word_count"] for k, v in meta["construct_validity"]["per_model"].items()}

    # Order high-disclosure partners descending by cost impact, then bottom two withholding partners below divider
    withholders = {"astra62", "kimi31"}
    transparent_sorted = sorted([p for p in cost_by_p if p not in withholders], key=lambda p: cost_by_p[p]["mean"], reverse=True)
    withholding_sorted = sorted([p for p in cost_by_p if p in withholders], key=lambda p: cost_by_p[p]["mean"], reverse=True)
    partners_ordered = [*transparent_sorted, *withholding_sorted]

    metrics_spec = [
        ("prompt_tokens", "Prompt tokens"),
        ("reasoning_tokens", "Reasoning tokens"),
        ("word_count", "Output words"),
        ("cost_usd", "API cost"),
    ]

    fig = plt.figure(figsize=(12.6, 5.0), facecolor=BG_WHITE)
    gs = fig.add_gridspec(1, 4, left=0.212, right=0.97, top=0.735, bottom=0.135, wspace=0.14)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(4)]

    # Y positions with generous vertical gap separating the 5 high-disclosure partners from the 2 withholding partners
    y_pos = np.array([0.40, 1.40, 2.40, 3.40, 4.40, 6.20, 7.20])

    bloat_export = []
    for p in partners_ordered:
        row_obj: dict[str, Any] = {"partner": p, "display": MODEL_FULL[p], "own_mean_words": round(own_words_by_p[p], 1)}
        for m_key, _ in metrics_spec:
            row_obj[m_key] = round(bt[m_key]["by_partner"][p]["mean"], 2)
        bloat_export.append(row_obj)

    for col_idx, (ax, (m_key, m_title)) in enumerate(zip(axes, metrics_spec, strict=True)):
        style_axes(ax, grid_axis="x")
        ax.spines["left"].set_visible(False)
        # Segmented 0% reference spine skips both group-header bands cleanly
        ax.vlines([0.0, 0.0], [0.02, 5.80], [4.80, 7.62], color=TEXT_BODY, linewidth=1.0, zorder=2)
        ax.axhline(5.12, color=MUTED_LIGHT, linestyle="--", linewidth=0.95, zorder=1)

        for row_idx, p in enumerate(partners_ordered):
            val = bt[m_key]["by_partner"][p]["mean"]
            yp = y_pos[row_idx]
            if p in withholders:
                bar_col = GREY_BAR_DARK
                txt_col = MUTED_DARK
                fw = "normal"
            elif val > 0.0:
                bar_col = VERMILION if val >= 8.0 else VERMILION_SOFT
                txt_col = VERMILION_DEEP if val >= 8.0 else TEXT_BODY
                fw = "bold" if val >= 8.0 else "normal"
            else:
                bar_col = TEAL if val <= -5.0 else TEAL_SOFT
                txt_col = TEAL_DEEP if val <= -5.0 else TEXT_BODY
                fw = "bold" if val <= -5.0 else "normal"

            ax.barh(yp, val, height=0.58, color=bar_col, alpha=0.92, edgecolor=BG_WHITE, linewidth=0.5, zorder=3)

            ha = "left" if val >= 0 else "right"
            dx = 1.6 if val >= 0 else -1.6
            val_lbl = "0.0%" if abs(val) < 0.05 else f"{val:+.1f}%"
            ax.text(
                val + dx,
                yp,
                val_lbl,
                ha=ha,
                va="center",
                fontsize=7.6,
                fontweight=fw,
                color=txt_col,
            )

        ax.set_xlim(-42, 52)
        ax.set_ylim(7.80, -0.78)
        ax.set_xticks([-20, 0, 20, 40])
        ax.set_xticklabels(["-20%", "0%", "+20%", "+40%"], fontsize=7.8)
        ax.set_title(m_title, loc="center", fontsize=9.6, fontweight="bold", pad=10, color=TEXT_DARK)

        if col_idx == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{MODEL_FULL[p]} ({own_words_by_p[p]:.0f}w)" for p in partners_ordered], fontsize=8.3, fontweight="bold", color=TEXT_DARK)
            ax.tick_params(axis="y", left=False, pad=6)
            ax.text(
                -41.0,
                -0.36,
                "High-disclosure partners",
                ha="left",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color=TEAL_DEEP,
                bbox={"facecolor": BG_WHITE, "edgecolor": "none", "pad": 1.0},
                clip_on=False,
                zorder=6,
            )
            ax.text(
                -41.0,
                5.36,
                "Under-informing partners (omit rules)",
                ha="left",
                va="top",
                fontsize=7.4,
                color=MUTED,
                bbox={"facecolor": BG_WHITE, "edgecolor": "none", "pad": 1.2},
                clip_on=False,
                zorder=6,
            )
        else:
            ax.set_yticks([])

    fig.text(
        0.585,
        0.035,
        "Within-task percentage change on teammate's subsequent turns",
        ha="center",
        va="bottom",
        fontsize=8.9,
        color=TEXT_BODY,
    )

    add_header(
        fig,
        title="Verbose partners inflate teammates' prompt tokens, reasoning and API bills",
        subtitle=(
            f"Within-task percentage change across {meta['data_inventory']['n_episodes_main']} "
            "live episodes (sorted by partner API cost within each group, own words/turn in parentheses)"
        ),
        left_x=0.04,
        title_fontsize=12.8,
    )

    payload = {
        "figure": stem,
        "sources": [str(META_PATH.relative_to(REPO_ROOT))],
        "n_episodes_main": meta["data_inventory"]["n_episodes_main"],
        "n_episode_side_rows": meta["bloat_tax_and_contagion"]["exposed_turns_primary"]["overall"]["n_rows"],
        "n_blocks": meta["bloat_tax_and_contagion"]["exposed_turns_primary"]["overall"]["n_blocks"],
        "by_partner": bloat_export,
    }
    save_figure_and_data(fig, stem, payload)


# =============================================================================
# FIGURE 7: Incident Triage Per-Incident Grid (Strategic Grey + Top Headers)
# =============================================================================
def build_fig7_triage_paralysis(meta: dict[str, Any]) -> None:
    """Figure 7: Mutual caution caused all 3 Incident Triage deadlocks with zero safety breaches."""
    stem = "fig7_triage_paralysis"
    triage_grid = meta["triage_paralysis_and_upsets"]["triage_7x7_grid"]
    deadlocks_n = meta["triage_paralysis_and_upsets"]["triage_below_099_inventory"]["astra60_astra62_00_deadlocks_count"]
    triage_order = ["astra62", "opus55", "neon4", "kimi31", "astra60", "muse", "qwen4"]
    seeds = (301, 302, 303, 304)

    per_pair_scores: dict[tuple[str, str], dict[int, float]] = {}
    triage_files = sorted(glob.glob(str(EPISODES_DIR / "triage__s30[1-4]__*.json")))
    if triage_files:
        for p_str in triage_files:
            ep = json.loads(Path(p_str).read_text(encoding="utf-8"))
            key = (ep["model_a"], ep["model_b"])
            per_pair_scores.setdefault(key, {})[int(ep["seed"])] = float(ep["score"])
    else:
        import pandas as pd

        ep_df = pd.read_parquet(REPO_ROOT / "data/v2/hf_export/episodes.parquet")
        sub = ep_df[(ep_df["task"] == "triage") & (ep_df["seed"].isin(seeds))]
        for row in sub.to_dict(orient="records"):
            key = (str(row["model_a"]), str(row["model_b"]))
            per_pair_scores.setdefault(key, {})[int(row["seed"])] = float(row["score"])

    if sum(len(v) for v in per_pair_scores.values()) != 160:
        raise ValueError("Expected 160 main-arm Incident Triage episodes")

    def tile_color(score: float) -> str:
        if score >= 0.99:
            return TILE_CLEAN
        if score >= 0.85:
            return TILE_MINOR
        if score >= 0.20:
            return AMBER
        return VERMILION

    fig = plt.figure(figsize=(9.7, 6.65), facecolor=BG_WHITE)
    ax = fig.add_axes([0.175, 0.06, 0.565, 0.69])
    ax.set_facecolor(BG_WHITE)

    n_m = len(triage_order)
    deadlock_pairs = {("astra60", "astra62"), ("astra62", "astra60")}
    opus_upset_pairs = {("astra62", "opus55"), ("astra60", "opus55")}

    # Wider inter-cell gap (0.82 outer cell inside 1.0 stride) so highlight frames never touch
    sub_offsets = [
        (-0.39, -0.39),  # incident 1 (top-left)
        (0.015, -0.39),  # incident 2 (top-right)
        (-0.39, 0.015),  # incident 3 (bottom-left)
        (0.015, 0.015),  # incident 4 (bottom-right)
    ]
    sub_w = 0.375
    sub_h = 0.375

    # Clean single shaded zone for the 3x3 partner-only unplayed block (no 9 repeated stamps)
    unplayed_zone = mpatches.Rectangle(
        (3.58, 3.58),
        2.84,
        2.84,
        facecolor=ROW_TINT,
        edgecolor=BORDER,
        linewidth=0.8,
        zorder=1,
    )
    ax.add_patch(unplayed_zone)
    ax.text(
        5.0,
        5.0,
        "Partner-only models\n(not paired with each other)",
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED,
        linespacing=1.3,
        zorder=2,
    )

    triage_export = []
    for i, r in enumerate(triage_order):
        for j, c in enumerate(triage_order):
            cell_meta = triage_grid["cells"].get(r, {}).get(c)
            if cell_meta is None:
                continue

            seed_map = per_pair_scores[(r, c)]
            mean_check = sum(seed_map[sd] for sd in seeds) / 4.0
            if abs(mean_check - cell_meta["mean_score"]) > 1e-4:
                raise ValueError(f"Triage score mismatch for {r} x {c}: {mean_check} vs {cell_meta['mean_score']}")

            bg = mpatches.Rectangle(
                (j - 0.41, i - 0.41),
                0.82,
                0.82,
                facecolor=BG_WHITE,
                edgecolor=BORDER,
                linewidth=0.55,
                zorder=2,
            )
            ax.add_patch(bg)

            for sd, (dx, dy) in zip(seeds, sub_offsets, strict=True):
                sc = seed_map[sd]
                col = tile_color(sc)
                tile = mpatches.Rectangle(
                    (j + dx, i + dy),
                    sub_w,
                    sub_h,
                    facecolor=col,
                    edgecolor=BG_WHITE,
                    linewidth=0.75,
                    zorder=3,
                )
                ax.add_patch(tile)
                if sc < 0.01:
                    ax.text(
                        j + dx + sub_w / 2,
                        i + dy + sub_h / 2,
                        "0.00",
                        ha="center",
                        va="center",
                        fontsize=6.3,
                        fontweight="bold",
                        color=BG_WHITE,
                        zorder=4,
                    )
                elif sc < 0.50:
                    ax.text(
                        j + dx + sub_w / 2,
                        i + dy + sub_h / 2,
                        "0.35",
                        ha="center",
                        va="center",
                        fontsize=5.6,
                        fontweight="bold",
                        color=BG_WHITE,
                        zorder=4,
                    )
                elif sc < 0.99:
                    ax.text(
                        j + dx + sub_w / 2,
                        i + dy + sub_h / 2,
                        f"0.{round(sc * 100)}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=TEXT_DARK,
                        zorder=4,
                    )

            if (r, c) in deadlock_pairs:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.445, i - 0.445),
                        0.89,
                        0.89,
                        fill=False,
                        edgecolor=TEXT_DARK,
                        linewidth=2.0,
                        zorder=6,
                    )
                )
            elif (r, c) in opus_upset_pairs:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.445, i - 0.445),
                        0.89,
                        0.89,
                        fill=False,
                        edgecolor=TEXT_BODY,
                        linestyle="--",
                        linewidth=1.5,
                        zorder=6,
                    )
                )

            triage_export.append(
                {
                    "investigator_role_a": r,
                    "investigator_display": MODEL_FULL[r],
                    "operator_role_b": c,
                    "operator_display": MODEL_FULL[c],
                    "seed_scores": {str(sd): seed_map[sd] for sd in seeds},
                    "mean_score": round(mean_check, 4),
                    "min_score": min(seed_map.values()),
                }
            )

    ax.axvline(3.5, color=MUTED_LIGHT, linestyle=":", linewidth=1.0, zorder=5)
    ax.axhline(3.5, color=MUTED_LIGHT, linestyle=":", linewidth=1.0, zorder=5)

    # Horizontal top Operator labels (zero diagonal text!)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlim(-0.55, n_m - 0.45)
    ax.set_ylim(n_m - 0.45, -0.55)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n_m))
    ax.set_xticklabels([MODEL_TWO_LINE[m] for m in triage_order], fontsize=8.0, color=TEXT_DARK, linespacing=1.12)
    ax.set_yticks(np.arange(n_m))
    ax.set_yticklabels([MODEL_FULL[m] for m in triage_order], fontsize=8.3, color=TEXT_DARK)
    ax.tick_params(length=0, pad=5)
    ax.set_xlabel("Operator (remediation commands)", fontsize=8.8, fontweight="bold", color=TEXT_BODY, labelpad=10)
    ax.set_ylabel("Investigator (diagnostics only)", fontsize=8.8, fontweight="bold", color=TEXT_BODY, labelpad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)

    leg_handles = [
        mpatches.Patch(facecolor=TILE_CLEAN, edgecolor=BORDER, label="1.00  Solved cleanly"),
        mpatches.Patch(facecolor=TILE_MINOR, edgecolor=BG_WHITE, label="0.90 to 0.95  Minor penalty"),
        mpatches.Patch(facecolor=AMBER, edgecolor=BG_WHITE, label="0.35  Partial (5 WAITs)"),
        mpatches.Patch(facecolor=VERMILION, edgecolor=BG_WHITE, label="0.00  Deadlock (6/6 WAIT)"),
        mpatches.Patch(facecolor="none", edgecolor=TEXT_DARK, linewidth=1.9, label="GPT-6 Astra / GPT-6.2 Astra\ndeadlocks"),
        mpatches.Patch(facecolor="none", edgecolor=TEXT_BODY, linestyle="--", linewidth=1.4, label="Claude Opus 5.5 chose this\n0.35 run over a 1.00 run"),
    ]
    ax.legend(
        handles=leg_handles,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        frameon=False,
        fontsize=7.9,
        labelspacing=0.95,
        handlelength=1.45,
        handleheight=1.1,
    )

    # Compact standalone 2x2 incident-order schematic under the legend
    ax.text(1.045, 0.34, "Cell layout (4 incidents)", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.6, fontweight="bold", color=MUTED, clip_on=False)
    for (kx, ky, klbl) in [(1.045, 0.275, "1"), (1.105, 0.275, "2"), (1.045, 0.215, "3"), (1.105, 0.215, "4")]:
        box = mpatches.Rectangle((kx, ky), 0.052, 0.052, transform=ax.transAxes, facecolor=TILE_CLEAN, edgecolor=BORDER, linewidth=0.8, clip_on=False)
        ax.add_patch(box)
        ax.text(kx + 0.026, ky + 0.026, klbl, transform=ax.transAxes, ha="center", va="center", fontsize=7.2, fontweight="bold", color=TEXT_DARK, clip_on=False)

    add_header(
        fig,
        title="Mutual caution caused all 3 Incident Triage deadlocks with zero safety breaches",
        subtitle="All 160 episodes (4 incidents per pair): hesitation (WAIT) drove 21 of 22 imperfect runs",
        left_x=0.04,
        title_fontsize=11.9,
    )

    payload = {
        "figure": stem,
        "sources": [str(META_PATH.relative_to(REPO_ROOT)), "data/v2/hf_export/episodes.parquet"],
        "models_order": triage_order,
        "astra60_astra62_zero_score_deadlocks": deadlocks_n,
        "cells": triage_export,
    }
    save_figure_and_data(fig, stem, payload)


# =============================================================================
# FIGURE A1 (Appendix A): Multivariate Peer-Preference Logits (Strategic Grey Nulls)
# =============================================================================
def build_figA1_logits(meta: dict[str, Any]) -> None:
    """Figure A1: Direct answers and concrete identifiers predict peer preference while warmth and length do not."""
    stem = "figA1_choice_logits"
    mv_all = meta["choice_logits"]["multivariate"]["all_non_aa_with_fe"]
    mv_ties = meta["choice_logits"]["multivariate"]["exact_ties_with_fe"]
    d_score_coef = mv_all["coefs"]["d_score"]["coef"]
    d_score_p_fmt = mv_all["coefs"]["d_score"]["p_fmt"]

    # Sorted cleanly: significant positive -> null predictors (descending) -> significant negative
    pred_spec = [
        ("d_answers_directly", "Answers directly"),
        ("d_wooden_legalistic", "Formal / legalistic style"),
        ("d_ids_per_100", "Identifiers per 100 words"),
        ("d_log_wc", "Log word count"),
        ("d_warmth", "Warmth"),
        ("d_skips_why", "Skips explaining why"),
        ("d_condescending_pedantic", "Condescending / pedantic"),
        ("d_clarity_resid", "Clarity (length-adjusted)"),
        ("d_s_conf", "Self-confusion"),
    ]

    fig = plt.figure(figsize=(10.0, 5.0), facecolor=BG_WHITE)
    ax = fig.add_axes([0.245, 0.14, 0.715, 0.645])
    style_axes(ax, grid_axis="x")

    y_l = np.arange(len(pred_spec))
    ax.axvline(0.0, color=TEXT_BODY, linestyle="--", linewidth=1.0, zorder=1)
    for i in range(len(pred_spec)):
        if i % 2 == 0:
            ax.axhspan(i - 0.45, i + 0.45, color=ROW_TINT, zorder=0)

    logits_export = []
    dy_all = -0.16
    dy_ties = +0.16
    for i, (key, lbl) in enumerate(pred_spec):
        c_all = mv_all["coefs"][key]
        c_tie = mv_ties["coefs"][key]

        b_all, se_all, p_all = c_all["coef"], c_all["se"], c_all["p"]
        b_tie, se_tie, p_tie = c_tie["coef"], c_tie["se"], c_tie["p"]
        sig_all = p_all < 0.05
        sig_tie = p_tie < 0.05

        col_all = SLATE_BLUE if sig_all else MUTED_LIGHT
        edge_all = SLATE_BLUE if sig_all else GREY_BAR_DARK
        col_tie = TEAL if sig_tie else MUTED_LIGHT
        edge_tie = TEAL if sig_tie else GREY_BAR_DARK

        ax.hlines(i + dy_all, b_all - 1.96 * se_all, b_all + 1.96 * se_all, color=col_all, linewidth=2.2 if sig_all else 1.5, zorder=3)
        ax.scatter(
            b_all,
            i + dy_all,
            marker="o",
            s=48 if sig_all else 36,
            facecolor=col_all if sig_all else BG_WHITE,
            edgecolor=edge_all,
            linewidth=1.5,
            zorder=4,
        )

        ax.hlines(i + dy_ties, b_tie - 1.96 * se_tie, b_tie + 1.96 * se_tie, color=col_tie, linewidth=2.2 if sig_tie else 1.5, zorder=3)
        ax.scatter(
            b_tie,
            i + dy_ties,
            marker="s",
            s=46 if sig_tie else 34,
            facecolor=col_tie if sig_tie else BG_WHITE,
            edgecolor=edge_tie,
            linewidth=1.5,
            zorder=4,
        )

        # Label each significant bar cleanly at its own tip
        if sig_all:
            x_end = b_all + 1.96 * se_all + 0.04 if b_all > 0 else b_all - 1.96 * se_all - 0.04
            ha = "left" if b_all > 0 else "right"
            ax.text(x_end, i + dy_all, f"{b_all:+.2f}", ha=ha, va="center", fontsize=7.9, fontweight="bold", color=SLATE_BLUE)
        if sig_tie:
            x_end = b_tie + 1.96 * se_tie + 0.04 if b_tie > 0 else b_tie - 1.96 * se_tie - 0.04
            ha = "left" if b_tie > 0 else "right"
            ax.text(x_end, i + dy_ties, f"{b_tie:+.2f}", ha=ha, va="center", fontsize=7.9, fontweight="bold", color=TEAL_DEEP)

        logits_export.append(
            {
                "predictor": key,
                "label": lbl,
                "all_live_choices_with_fe": c_all,
                "exact_ties_with_fe": c_tie,
            }
        )

    ax.set_yticks(y_l)
    ax.set_yticklabels([lbl for _, lbl in pred_spec], fontsize=8.5, color=TEXT_DARK)
    for tick_lbl, (key, _) in zip(ax.get_yticklabels(), pred_spec, strict=True):
        if mv_all["coefs"][key]["p"] < 0.05 or mv_ties["coefs"][key]["p"] < 0.05:
            tick_lbl.set_fontweight("bold")
        else:
            tick_lbl.set_color(MUTED_DARK)
    ax.invert_yaxis()
    ax.set_xlim(-1.05, 1.32)
    ax.set_ylim(len(pred_spec) - 0.35, -0.65)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("Standardised log-odds effect per 1-SD increase (95% CI)")

    # Sleek borderless 1-line inline key replacing boxed legend and duplicate significance note
    ax.scatter([0.015], [1.055], transform=ax.transAxes, clip_on=False, marker="o", s=38, facecolor=SLATE_BLUE, edgecolor=SLATE_BLUE)
    ax.text(0.038, 1.032, f"All live choices (N={mv_all['n_choices']:,}, p < 0.05)", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=SLATE_BLUE)
    ax.scatter([0.415], [1.055], transform=ax.transAxes, clip_on=False, marker="s", s=36, facecolor=TEAL, edgecolor=TEAL)
    ax.text(0.438, 1.032, f"Exact score ties (N={mv_ties['n_choices']:,}, p < 0.05)", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=TEAL_DEEP)
    ax.scatter([0.825], [1.055], transform=ax.transAxes, clip_on=False, marker="o", s=34, facecolor=BG_WHITE, edgecolor=GREY_BAR_DARK, linewidth=1.4)
    ax.text(0.848, 1.032, "p ≥ 0.05 (null)", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, color=MUTED_DARK)

    add_header(
        fig,
        title="Direct answers and concrete identifiers predict peer preference while warmth and length do not",
        subtitle=(
            f"Cluster-robust log-odds effects (partner fixed effects), omitting task-score delta ({d_score_coef:+.2f}, p {d_score_p_fmt}) for scale"
        ),
        left_x=0.04,
        title_fontsize=12.0,
    )

    payload = {
        "figure": stem,
        "sources": [str(META_PATH.relative_to(REPO_ROOT))],
        "n_choices_all_live": mv_all["n_choices"],
        "n_choices_exact_ties": mv_ties["n_choices"],
        "n_clusters_all_live": mv_all["n_clusters"],
        "n_clusters_exact_ties": mv_ties["n_clusters"],
        "omitted_d_score_all_live": mv_all["coefs"]["d_score"],
        "predictors": logits_export,
    }
    save_figure_and_data(fig, stem, payload)


def parse_args() -> argparse.Namespace:
    """Parse optional CLI arguments for paper-themed vector PDF export."""
    parser = argparse.ArgumentParser(
        description="Generate publication figures (default: web SVG/PNG + JSON; --pdf-dir: PDF)."
    )
    parser.add_argument(
        "--theme",
        choices=("web", "paper"),
        default="web",
        help="Palette theme: 'web' (#F1EADC warm ground) or 'paper' (#FFFFFF white ground).",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Export vector PDFs into this directory without modifying docs/ or site/.",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=None,
        help="Optional directory containing Spectral .ttf/.otf files to register with Matplotlib.",
    )
    args = parser.parse_args()
    if args.theme == "paper" and args.pdf_dir is None:
        parser.error("--theme paper requires --pdf-dir so warm-ground web assets are untouched.")
    return args


def main() -> None:
    """Load frozen study artifacts and generate all 8 publication figures and companion datasets."""
    args = parse_args()
    apply_runtime_options(args.theme, args.pdf_dir)
    configure_editorial_style(theme=args.theme, font_dir=args.font_dir)

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    exp1 = json.loads(EXP1_PATH.read_text(encoding="utf-8"))
    exp1_rows = load_deduped_exp1_results()
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    exp2 = json.loads(EXP2_PATH.read_text(encoding="utf-8"))

    build_fig1(meta)
    build_fig2_clarity_slopegraph(meta)
    build_fig3_causal_dissociation(exp2)
    build_fig4_dropout(exp1, gate, exp1_rows)
    build_fig5_heatmap(exp1)
    build_fig6_bloat_tax(meta)
    build_fig7_triage_paralysis(meta)
    build_figA1_logits(meta)

    stems = [
        "fig1_eval_blindness_within_gpt62",
        "fig2_clarity_length_bias",
        "fig3_causal_double_dissociation",
        "fig4_dropout_sender_and_ladder",
        "fig5_sender_x_receiver_heatmap",
        "fig6_compute_bloat_tax",
        "fig7_triage_paralysis",
        "figA1_choice_logits",
    ]
    if PDF_OUTPUT_DIR is not None:
        for s in stems:
            pdf = PDF_OUTPUT_DIR / f"{s}.pdf"
            if not pdf.exists() or pdf.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty PDF output: {pdf}")
            print(f"OK {s}: pdf={pdf.stat().st_size}B")
        return

    for s in stems:
        svg = FIG_DIR / f"{s}.svg"
        png = FIG_DIR / f"{s}.png"
        js = DATA_DIR / f"{s}.json"
        for p in (svg, png, js):
            if not p.exists() or p.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty output: {p}")
        print(f"OK {s}: svg={svg.stat().st_size}B png={png.stat().st_size}B json={js.stat().st_size}B")


if __name__ == "__main__":
    main()
