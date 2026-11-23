"""Shared cinematic theme, color system, and reusable Manim helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from manim import (
    BLACK,
    BLUE,
    DOWN,
    DR,
    FadeIn,
    GRAY,
    LEFT,
    ORIGIN,
    RIGHT,
    Text,
    MathTex,
    Tex,
    UP,
    UL,
    UR,
    VGroup,
    Dot,
    Line,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineSummary, run_math_pipeline

# ---------------------------------------------------------------------------
# Color Constitution (matched strictly to project specification)
# ---------------------------------------------------------------------------
BG_NAVY = "#050914"
BG_PANEL = "#0B1326"
BG_PANEL_BORDER = "#1D3557"

COLOR_ORIGINAL = "#E6F9FF"       # White/cyan for original image & matrix
COLOR_CYAN = "#22D3EE"           # Cyan highlights & diagonal matrix
COLOR_EIGENVECTOR = "#3B82F6"    # Electric blue for eigenvectors
COLOR_EIGENVALUE = "#A855F7"     # Purple for eigenvalues
COLOR_DIAGONAL = "#06B6D4"       # Cyan for diagonal matrix Lambda / Sigma
COLOR_DISCARDED = "#4B5563"      # Muted grey for discarded components
COLOR_RECON = "#60A5FA"          # Bright blue / white for reconstructed image
COLOR_WHITE = "#F8FAFC"
COLOR_SUBTEXT = "#94A3B8"
COLOR_ACCENT_GOLD = "#FBBF24"


def configure_scene_canvas() -> None:
    """Apply consistent 16:9 dark navy documentary aesthetic."""
    config.background_color = BG_NAVY


def get_pipeline_data() -> PipelineSummary:
    """Run or refresh the verified mathematical computation pipeline."""
    return run_math_pipeline(project_root=PROJECT_ROOT)


def create_ambient_particles(
    count: int = 38,
    seed: int = 42,
    x_range: tuple[float, float] = (-7.0, 7.0),
    y_range: tuple[float, float] = (-3.9, 3.9),
) -> VGroup:
    """Create minimal atmospheric cyan/blue particles for depth."""
    rng = np.random.default_rng(seed)
    particles = VGroup()
    palette = [COLOR_CYAN, COLOR_EIGENVECTOR, COLOR_EIGENVALUE, COLOR_WHITE]
    for i in range(count):
        x = float(rng.uniform(*x_range))
        y = float(rng.uniform(*y_range))
        radius = float(rng.uniform(0.014, 0.042))
        opacity = float(rng.uniform(0.10, 0.35))
        color = palette[i % len(palette)]
        dot = Dot(point=[x, y, 0], radius=radius, color=color, fill_opacity=opacity)
        particles.add(dot)
    return particles


def create_glow_frame(
    target,
    color: str = COLOR_CYAN,
    buff: float = 0.06,
    corner_radius: float = 0.08,
) -> VGroup:
    """Build a layered luminous frame around an image or matrix."""
    inner = RoundedRectangle(
        width=target.width + 2 * buff,
        height=target.height + 2 * buff,
        corner_radius=corner_radius,
        stroke_color=color,
        stroke_width=2.2,
        fill_opacity=0,
    ).move_to(target)

    mid = RoundedRectangle(
        width=target.width + 2 * buff + 0.08,
        height=target.height + 2 * buff + 0.08,
        corner_radius=corner_radius + 0.02,
        stroke_color=color,
        stroke_width=5.0,
        stroke_opacity=0.22,
        fill_opacity=0,
    ).move_to(target)

    outer = RoundedRectangle(
        width=target.width + 2 * buff + 0.18,
        height=target.height + 2 * buff + 0.18,
        corner_radius=corner_radius + 0.04,
        stroke_color=color,
        stroke_width=10.0,
        stroke_opacity=0.08,
        fill_opacity=0,
    ).move_to(target)

    return VGroup(outer, mid, inner)


def create_scene_header(
    chapter_label: str,
    title_text: str,
    accent_color: str = COLOR_CYAN,
) -> VGroup:
    """Create a documentary-style top-left chapter header."""
    kicker = Text(
        chapter_label.upper(),
        font_size=16,
        color=accent_color,
        weight="BOLD",
    )
    title = Text(
        title_text,
        font_size=30,
        color=COLOR_WHITE,
        weight="SEMIBOLD",
    )
    title.next_to(kicker, DOWN, aligned_edge=LEFT, buff=0.08)
    rule = Line(
        start=ORIGIN,
        end=RIGHT * max(title.width, 3.2),
        stroke_color=accent_color,
        stroke_width=1.8,
        stroke_opacity=0.65,
    )
    rule.next_to(title, DOWN, aligned_edge=LEFT, buff=0.10)
    group = VGroup(kicker, title, rule)
    group.to_corner(UL, buff=0.45)
    return group


def create_metric_badge(
    label: str,
    value: str,
    accent_color: str = COLOR_CYAN,
    width: float = 3.2,
    height: float = 0.85,
) -> VGroup:
    """Create a sleek dark-glass metric readout card."""
    box = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.1,
        fill_color=BG_PANEL,
        fill_opacity=0.88,
        stroke_color=accent_color,
        stroke_width=1.5,
        stroke_opacity=0.65,
    )
    lbl = Text(label.upper(), font_size=14, color=COLOR_SUBTEXT)
    val = Text(value, font_size=24, color=COLOR_WHITE, weight="BOLD")
    lbl.move_to(box.get_top() + DOWN * 0.22)
    val.move_to(box.get_bottom() + UP * 0.28)
    return VGroup(box, lbl, val)


def create_subtitle_caption(text: str, color: str = COLOR_ORIGINAL) -> VGroup:
    """Create a subtle documentary lower-third caption bar."""
    txt = Text(text, font_size=22, color=color)
    bg = RoundedRectangle(
        width=txt.width + 0.65,
        height=txt.height + 0.28,
        corner_radius=0.12,
        fill_color=BG_NAVY,
        fill_opacity=0.78,
        stroke_color=BG_PANEL_BORDER,
        stroke_width=1.0,
    )
    bg.move_to(txt)
    group = VGroup(bg, txt)
    group.to_edge(DOWN, buff=0.32)
    return group
