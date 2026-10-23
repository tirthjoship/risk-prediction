"""Visual identity and HTML helpers for the layered decision brief.

Every tab has three layers: a decision layer (the bottom line), an evidence
layer (charts and tables in bordered cards, each with a one sentence
takeaway) and a method layer (tooltips and a collapsed "How we know" panel).

The builders here are pure functions that return HTML strings, so they can be
unit tested without Streamlit. Every inserted string is escaped.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from html import escape
from typing import Any

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY = "#1F3A5F"
AMBER = "#B7791F"
INK = "#0D0F12"
SLATE = "#5C6B7A"
CARD_BORDER = "#EAECEF"
PAPER = "#FFFFFF"
RED = "#B3261E"
GREEN = "#2F7D4F"

FONT_BODY = "Inter, sans-serif"
FONT_DISPLAY = "Space Grotesk, sans-serif"
FONT_MONO = "IBM Plex Mono, monospace"

# tone -> (text color, soft background, edge color)
TONES: dict[str, tuple[str, str, str]] = {
    "problem": (RED, "#F9E2E0", RED),
    "caution": (AMBER, "#F8EDD6", AMBER),
    "good": (GREEN, "#E3F3E9", GREEN),
    "neutral": (NAVY, "#E8EDF4", NAVY),
}


def tone_colors(tone: str) -> tuple[str, str, str]:
    """Return (text, soft background, edge) colors for a tone; unknown is neutral."""
    return TONES.get(tone, TONES["neutral"])


def tone_for_probability(probability: float) -> str:
    """Map a late probability to a tone using the same bands as the gauge."""
    if probability > 0.70:
        return "problem"
    if probability > 0.40:
        return "caution"
    return "good"


def _e(text: object) -> str:
    return escape(str(text), quote=True)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --navy: #1F3A5F; --amber: #B7791F; --ink: #0D0F12; --slate: #5C6B7A;
  --line: #EAECEF; --soft: #F6F7F9;
  --red: #B3261E; --green: #2F7D4F;
}
.stApp, [data-testid="stAppViewContainer"] { background: #FFFFFF; }
html, body, p, li, label, .stMarkdown, [data-testid="stWidgetLabel"] {
  font-family: 'Inter', system-ui, sans-serif;
}
h1, h2, h3, h4, h5 {
  font-family: 'Space Grotesk', system-ui, sans-serif !important;
  letter-spacing: -0.02em; color: var(--ink);
}
.block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1240px; }
[data-testid="stVerticalBlockBorderWrapper"] { overflow: visible; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  padding: 10px 18px; font-family: 'Space Grotesk', system-ui, sans-serif;
  font-weight: 600; color: var(--slate);
}
.stTabs [aria-selected="true"] { color: var(--navy); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--amber); height: 3px; }

/* Bordered cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--line) !important; border-radius: 12px !important;
}

/* Header */
.ld-header { margin: 0 0 18px; }
.ld-eyebrow {
  font: 600 11.5px 'IBM Plex Mono', monospace; letter-spacing: .16em;
  text-transform: uppercase; color: var(--amber);
}
.ld-headline {
  font: 700 38px/1.1 'Space Grotesk', system-ui, sans-serif; letter-spacing: -0.025em;
  color: var(--navy); margin: 8px 0 8px;
}
.ld-summary { font-size: 15.5px; line-height: 1.5; color: #3A4047; max-width: 780px; margin: 0 0 14px; }
.ld-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.ld-chip {
  font: 500 12px 'IBM Plex Mono', monospace; color: #48505A; background: var(--soft);
  border: 1px solid var(--line); border-radius: 999px; padding: 4px 12px; position: relative;
}
.ld-chip b { color: var(--ink); font-weight: 600; }

/* Tooltip */
.ld-i {
  display: inline-flex; align-items: center; justify-content: center;
  width: 15px; height: 15px; margin-left: 6px; border-radius: 50%;
  border: 1px solid #C2C7CD; color: #8B919A; font: 600 10px 'Inter', sans-serif;
  cursor: help; position: relative; vertical-align: middle;
}
.ld-i:hover { border-color: var(--navy); color: var(--navy); }
.ld-i:hover::after {
  content: attr(data-def); position: absolute; left: 0; top: 1.8em; z-index: 60;
  background: #11161D; color: #FFFFFF; font: 400 12px/1.5 'Inter', sans-serif;
  width: 280px; padding: 10px 12px; border-radius: 9px; white-space: normal;
  text-align: left; box-shadow: 0 8px 24px rgba(0, 0, 0, .28); letter-spacing: 0;
}

/* Bottom line */
.ld-bl {
  border: 1px solid var(--line); border-left: 5px solid var(--edge, var(--navy));
  border-radius: 12px; padding: 18px 22px 16px; margin: 6px 0 18px;
  background: linear-gradient(180deg, #FAFBFD, #FFFFFF);
}
.ld-bl-label {
  font: 600 11px 'IBM Plex Mono', monospace; letter-spacing: .16em;
  text-transform: uppercase; color: var(--slate);
}
.ld-bl-q { font: 600 21px/1.25 'Space Grotesk', system-ui, sans-serif; margin: 6px 0 10px; color: var(--ink); }
.ld-verdict {
  display: inline-block; font: 600 13.5px 'IBM Plex Mono', monospace;
  padding: 5px 14px; border-radius: 999px; margin-bottom: 10px;
  color: var(--tone); background: var(--soft-bg);
}
.ld-verdict.small { font-size: 12px; padding: 3px 11px; margin: 2px 0 0; }
.ld-bl p { margin: 0 0 6px; font-size: 14.5px; line-height: 1.55; color: #2A2F36; max-width: 860px; }
.ld-bl-foot {
  display: flex; flex-wrap: wrap; gap: 10px 28px; margin-top: 12px;
  padding-top: 12px; border-top: 1px dashed var(--line);
}
.ld-bl-foot .lab {
  font: 600 10.5px 'IBM Plex Mono', monospace; letter-spacing: .14em;
  text-transform: uppercase; color: var(--slate); display: block; margin-bottom: 2px;
}
.ld-bl-foot .val { font-size: 13.5px; color: var(--ink); font-weight: 500; }
.ld-bl-foot .act .val { color: var(--navy); font-weight: 600; }

/* Card heading */
.ld-card-head { margin: 0 0 4px; }
.ld-card-title { font: 600 16px 'Space Grotesk', system-ui, sans-serif; color: var(--ink); }
.ld-takeaway { font-size: 13.5px; color: var(--slate); margin: 2px 0 6px; line-height: 1.45; }

/* Tiles */
.ld-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 4px 0 10px; }
.ld-tile { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; background: #FFFFFF; position: relative; }
.ld-tile .lab { font-size: 12px; color: var(--slate); }
.ld-tile .num {
  font: 600 27px/1.15 'Space Grotesk', system-ui, sans-serif; margin-top: 2px;
  color: var(--tone, var(--ink)); letter-spacing: -0.02em;
}
.ld-tile .sub { font: 500 11.5px 'IBM Plex Mono', monospace; color: var(--slate); margin-top: 2px; }
.ld-tiles.small .num { font-size: 20px; }
.ld-tiles.stack { grid-template-columns: 1fr; gap: 8px; }

/* Source strip */
.ld-source {
  font-size: 12.5px; color: #48505A; background: var(--soft); border: 1px solid var(--line);
  border-left: 3px solid var(--navy); border-radius: 8px; padding: 8px 14px; margin: 0 0 14px;
}
.ld-source.warn { border-left-color: var(--amber); }

/* Sidebar */
[data-testid="stSidebar"] { background: #F6F7F9; border-right: 1px solid var(--line); }
.ld-side-title {
  font: 600 11px 'IBM Plex Mono', monospace; letter-spacing: .16em;
  text-transform: uppercase; color: var(--amber); margin: 6px 0 8px;
}
.ld-side-foot { font-size: 12px; color: var(--slate); margin-top: 8px; }
.ld-side-foot a { color: var(--navy); }

/* Buttons and metrics */
.stButton > button[kind="primary"] {
  background: var(--navy); border-color: var(--navy); font-weight: 600;
}
.stButton > button[kind="primary"]:hover { background: #162B48; border-color: #162B48; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; font-weight: 600; }
[data-testid="stExpander"] { border-color: var(--line); border-radius: 10px; }
.js-plotly-plot, .plot-container { max-width: 100% !important; }

/* Overview board */
.ld-tiles.four { grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 6px 0 12px; }
.ld-tiles.four .num { font-size: 26px; }
.ov-head { min-height: 5.5em; }
.ov-foot { font-size: 12px; color: var(--slate); margin: 10px 0 0; }
[data-testid="stSelectbox"] label p { font-size: 12.5px; color: var(--slate); }

.author-foot {
  margin-top: 28px; padding-top: 12px; border-top: 1px solid var(--line);
  font: 400 12px/1.6 'Inter', system-ui, sans-serif; color: var(--slate);
}
.author-foot a { color: var(--navy); text-decoration: none; border-bottom: 1px solid currentColor; }
.author-foot a:hover { opacity: .8; }

@media (max-width: 900px) {
  .ld-tiles.four { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ov-head { min-height: 0; }
}

@media (max-width: 720px) {
  .ld-headline { font-size: 28px; }
  .ld-bl-q { font-size: 18px; }
}
</style>
"""


def info(text: str) -> str:
    """Return a small circled i that shows text on hover."""
    return f'<span class="ld-i" data-def="{_e(text)}">i</span>'


def chip(label: str, tip: str | None = None) -> str:
    """Return one mono chip, with an optional hover tooltip."""
    tip_html = info(tip) if tip else ""
    return f'<span class="ld-chip">{_e(label)}{tip_html}</span>'


def header(
    eyebrow: str,
    headline: str,
    summary: str,
    chips: Sequence[tuple[str, str | None]] = (),
) -> str:
    """Return the app header: eyebrow, question headline, summary and chips."""
    chips_html = "".join(chip(label, tip) for label, tip in chips)
    chips_block = f'<div class="ld-chips">{chips_html}</div>' if chips_html else ""
    return (
        '<div class="ld-header">'
        f'<div class="ld-eyebrow">{_e(eyebrow)}</div>'
        f'<div class="ld-headline">{_e(headline)}</div>'
        f'<p class="ld-summary">{_e(summary)}</p>'
        f"{chips_block}</div>"
    )


def bottom_line(
    question: str,
    answer: str,
    tone: str,
    sentences: Sequence[str],
    act: str | None = None,
    confidence: str | None = None,
    confidence_why: str | None = None,
    would_change: str | None = None,
) -> str:
    """Return the decision layer box: question, colored answer chip, plain sentences.

    The footer shows what to do, how confident we are (with a tooltip for why)
    and what would change the answer. Missing parts are left out.
    """
    text, soft, edge = tone_colors(tone)
    paragraphs = "".join(f"<p>{_e(s)}</p>" for s in sentences)
    foot_parts: list[str] = []
    if act:
        foot_parts.append(
            f'<div class="act"><span class="lab">What to do</span>'
            f'<span class="val">{_e(act)}</span></div>'
        )
    if confidence:
        why = info(confidence_why) if confidence_why else ""
        foot_parts.append(
            f'<div><span class="lab">Confidence</span>'
            f'<span class="val">{_e(confidence)}{why}</span></div>'
        )
    if would_change:
        foot_parts.append(
            f'<div><span class="lab">What would change this</span>'
            f'<span class="val">{_e(would_change)}</span></div>'
        )
    foot = f'<div class="ld-bl-foot">{"".join(foot_parts)}</div>' if foot_parts else ""
    return (
        f'<div class="ld-bl" style="--edge:{edge}">'
        '<div class="ld-bl-label">The bottom line</div>'
        f'<div class="ld-bl-q">{_e(question)}</div>'
        f'<div class="ld-verdict" style="--tone:{text};--soft-bg:{soft}">{_e(answer)}</div>'
        f"{paragraphs}{foot}</div>"
    )


def card_heading(
    title: str, takeaway: str | None = None, tip: str | None = None
) -> str:
    """Return the heading block of a card: title, optional tooltip and takeaway."""
    tip_html = info(tip) if tip else ""
    take = f'<div class="ld-takeaway">{_e(takeaway)}</div>' if takeaway else ""
    return (
        '<div class="ld-card-head">'
        f'<div class="ld-card-title">{_e(title)}{tip_html}</div>{take}</div>'
    )


def tile(
    label: str,
    value: str,
    tone: str = "neutral",
    sub: str | None = None,
    tip: str | None = None,
) -> str:
    """Return one big-number tile whose number is colored by tone."""
    color = tone_colors(tone)[0] if tone != "neutral" else INK
    sub_html = f'<div class="sub">{_e(sub)}</div>' if sub else ""
    tip_html = info(tip) if tip else ""
    return (
        f'<div class="ld-tile" style="--tone:{color}">'
        f'<div class="lab">{_e(label)}{tip_html}</div>'
        f'<div class="num">{_e(value)}</div>{sub_html}</div>'
    )


def tiles(items: Sequence[str], variant: str = "") -> str:
    """Wrap pre-built tiles in a responsive grid. variant: '', 'small' or 'stack'."""
    return f'<div class="ld-tiles {_e(variant)}">{"".join(items)}</div>'


def tag(text: str, tone: str = "neutral") -> str:
    """Return a small colored status tag."""
    color, soft, _ = tone_colors(tone)
    return (
        f'<span class="ld-verdict small" style="--tone:{color};--soft-bg:{soft}">'
        f"{_e(text)}</span>"
    )


def source_note(text: str, warn: bool = False) -> str:
    """Return the one line strip that says which data is on screen."""
    cls = "ld-source warn" if warn else "ld-source"
    return f'<div class="{cls}">{_e(text)}</div>'


def side_title(text: str) -> str:
    """Return a small sidebar section label."""
    return f'<div class="ld-side-title">{_e(text)}</div>'


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------
def plotly_layout(**overrides: Any) -> dict[str, Any]:
    """Return the shared Plotly layout: fonts, light gridlines, no chart title."""
    base: dict[str, Any] = {
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PAPER,
        "font": {"family": FONT_BODY, "color": INK, "size": 12},
        "title": {"text": None},
        "hoverlabel": {"font": {"family": FONT_BODY, "size": 12}},
        "xaxis": {
            "gridcolor": "#EEF0F3",
            "zeroline": False,
            "linecolor": CARD_BORDER,
            "title": {"font": {"color": SLATE, "size": 12}},
        },
        "yaxis": {
            "gridcolor": "#EEF0F3",
            "zeroline": False,
            "linecolor": CARD_BORDER,
            "title": {"font": {"color": SLATE, "size": 12}},
        },
    }
    base.update(overrides)
    return base


def style_figure(fig: go.Figure) -> go.Figure:
    """Apply the shared layout to a figure without touching its margins or sizes."""
    fig.update_layout(**plotly_layout())
    return fig


# ---------------------------------------------------------------------------
# Streamlit glue (thin wrappers around the pure builders above)
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Inject fonts and the stylesheet once per run."""
    st.markdown(CSS, unsafe_allow_html=True)


def author_footer(note: str = "") -> str:
    """Slim author credit line for the bottom of the page (static strings only)."""
    tail = f" · {note}" if note else ""
    return (
        '<div class="author-foot">Built by Tirth Joshi · '
        '<a href="https://github.com/tirthjoship/risk-prediction" target="_blank" rel="noopener">Source code on GitHub</a> · '
        '<a href="https://linkedin.com/in/tirthjoshi" target="_blank" rel="noopener">LinkedIn</a>'
        f"{tail}</div>"
    )


def render(html: str) -> None:
    """Render a builder's HTML."""
    st.markdown(html, unsafe_allow_html=True)


@contextmanager
def card(
    title: str, takeaway: str | None = None, tip: str | None = None
) -> Iterator[None]:
    """Bordered card with a heading and a one sentence takeaway."""
    with st.container(border=True):
        render(card_heading(title, takeaway, tip))
        yield
