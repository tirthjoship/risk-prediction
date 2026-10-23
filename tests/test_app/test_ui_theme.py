"""Tests for the pure HTML helpers in the decision brief theme."""

import plotly.graph_objects as go

from app.components import ui_theme
from app.components.cohort_viz import segment_bottom_line
from app.components.dataset_overview import dataset_bottom_line
from app.components.model_comparison import model_bottom_line


def test_tone_colors_known_and_unknown() -> None:
    assert ui_theme.tone_colors("problem")[0] == ui_theme.RED
    assert ui_theme.tone_colors("caution")[0] == ui_theme.AMBER
    assert ui_theme.tone_colors("good")[0] == ui_theme.GREEN
    assert ui_theme.tone_colors("nonsense") == ui_theme.TONES["neutral"]


def test_tone_for_probability_bands() -> None:
    assert ui_theme.tone_for_probability(0.71) == "problem"
    assert ui_theme.tone_for_probability(0.70) == "caution"
    assert ui_theme.tone_for_probability(0.41) == "caution"
    assert ui_theme.tone_for_probability(0.40) == "good"


def test_header_escapes_and_renders_chips() -> None:
    html = ui_theme.header(
        "EYEBROW <b>", "Why?", "A & B", chips=[("65,752 orders", None), ("F1", "tip")]
    )
    assert "&lt;b&gt;" in html and "A &amp; B" in html
    assert html.count('class="ld-chip"') == 2
    assert 'data-def="tip"' in html


def test_header_without_chips_has_no_chip_row() -> None:
    assert "ld-chips" not in ui_theme.header("e", "h", "s")


def test_bottom_line_escapes_every_field() -> None:
    html = ui_theme.bottom_line(
        question="<q>",
        answer="<a>",
        tone="problem",
        sentences=["<s1>", "s2"],
        act="<act>",
        confidence="<c>",
        confidence_why='"why"',
        would_change="<w>",
    )
    for raw in ("<q>", "<a>", "<s1>", "<act>", "<c>", "<w>"):
        assert raw not in html
    assert "&quot;why&quot;" in html
    assert ui_theme.RED in html
    assert "What to do" in html and "Confidence" in html


def test_bottom_line_minimal_has_no_footer() -> None:
    html = ui_theme.bottom_line("q", "a", "neutral", ["s"])
    assert "ld-bl-foot" not in html


def test_tile_tone_and_escape() -> None:
    html = ui_theme.tile("<L>", "5%", tone="problem", sub="<sub>", tip="t")
    assert "&lt;L&gt;" in html and "&lt;sub&gt;" in html
    assert ui_theme.RED in html
    assert ui_theme.INK in ui_theme.tile("L", "1")
    assert 'class="ld-tiles small"' in ui_theme.tiles(["x"], variant="small")


def test_card_heading_and_tag_and_notes() -> None:
    assert "&lt;t&gt;" in ui_theme.card_heading("<t>", "take", "tip")
    assert "ld-takeaway" not in ui_theme.card_heading("t")
    assert ui_theme.AMBER in ui_theme.tag("x", "caution")
    assert "warn" in ui_theme.source_note("n", warn=True)
    assert "warn" not in ui_theme.source_note("n")
    assert "&lt;" in ui_theme.side_title("<")


def test_plotly_layout_overrides_and_style_figure() -> None:
    layout = ui_theme.plotly_layout(height=300)
    assert layout["height"] == 300
    assert layout["font"]["family"].startswith("Inter")
    fig = ui_theme.style_figure(go.Figure(go.Bar(x=[1], y=["a"])))
    assert fig.layout.paper_bgcolor == ui_theme.PAPER
    assert fig.layout.xaxis.gridcolor == "#EEF0F3"


def test_css_declares_fonts_and_accent() -> None:
    assert "Space+Grotesk" in ui_theme.CSS and "IBM+Plex+Mono" in ui_theme.CSS
    assert ui_theme.NAVY.lower() in ui_theme.CSS.lower()


def test_model_bottom_line_barely_and_yes() -> None:
    barely = model_bottom_line(0.6531, 0.6648, True)
    assert "Barely: +0.012 F1" in barely
    assert "73% of SHAP" in barely
    assert "shipping-mode rule" in barely
    behind = model_bottom_line(0.70, 0.646, False)
    assert "No: -0.054 F1" in behind
    assert ui_theme.AMBER in behind
    yes = model_bottom_line(0.60, 0.70, False)
    assert "Yes: +0.100 F1" in yes
    assert "73%" not in yes


def test_segment_bottom_line_picks_riskiest() -> None:
    segments = [
        {
            "size": 100,
            "late_rate": 0.40,
            "avg_scheduled_days": 4.0,
            "dominant_shipping_mode": "Standard Class",
        },
        {
            "size": 50,
            "late_rate": 0.80,
            "avg_scheduled_days": 1.0,
            "dominant_shipping_mode": "Second Class",
        },
    ]
    html = segment_bottom_line(segments, True)
    assert "Second Class segment: 50 orders at 80.0% late" in html
    assert "Prioritize orders in this segment" in html
    low = segment_bottom_line([{**segments[0], "late_rate": 0.30}], False)
    assert "No segment needs urgent action." in low


def test_dataset_bottom_line_narrow_and_wide() -> None:
    modes = {"First Class": 0.953, "Standard Class": 0.381}
    narrow = dataset_bottom_line({"A": 0.495, "B": 0.576}, modes, True)
    assert "Barely: every region is 49.5% to 57.6% late" in narrow
    assert "Focus on shipping mode, not geography." in narrow
    wide = dataset_bottom_line({"A": 0.2, "B": 0.7}, modes, False)
    assert "Too few orders per region to tell" in wide
    full_wide = dataset_bottom_line({"A": 0.2, "B": 0.7}, modes, True)
    assert "Yes:" in full_wide


def test_author_footer_credit_line() -> None:
    out = ui_theme.author_footer("A note.")
    assert "Built by Tirth Joshi" in out and out.endswith("A note.</div>")
    assert out.count('target="_blank" rel="noopener"') == 2
    assert "https://github.com/tirthjoship/risk-prediction" in out
    assert "https://linkedin.com/in/tirthjoshi" in out
