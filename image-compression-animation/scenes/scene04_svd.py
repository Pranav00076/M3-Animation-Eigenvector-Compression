"""Scene 4: Singular Value Decomposition — SVD (2:05 - 3:10)

Introduces A = U Sigma V^T for rectangular image matrices, animates U, Sigma, V^T
separately, proves lambda_i = sigma_i^2, animates the real descending singular value
spectrum from our photograph, and contrasts dominant cyan/purple components against
muted grey discarded components.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Axes,
    Brace,
    Create,
    FadeIn,
    FadeOut,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    Write,
    smooth,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenes.theme import (
    BG_NAVY,
    BG_PANEL,
    COLOR_CYAN,
    COLOR_DIAGONAL,
    COLOR_DISCARDED,
    COLOR_EIGENVALUE,
    COLOR_EIGENVECTOR,
    COLOR_ORIGINAL,
    COLOR_SUBTEXT,
    COLOR_WHITE,
    configure_scene_canvas,
    create_ambient_particles,
    create_scene_header,
    get_pipeline_data,
)


class Scene04SVD(Scene):
    """2:05 - 3:10 | A = U Sigma V^T, lambda_i = sigma_i^2, and singular spectrum."""

    def construct(self) -> None:
        configure_scene_canvas()
        data = get_pipeline_data()
        m, n = data.image_data.height, data.image_data.width
        r = data.svd_result.full_rank
        s_vals = data.svd_result.singular_values
        cum_energy = data.svd_result.cumulative_energy_ratio

        particles = create_ambient_particles(count=36, seed=404)
        self.add(particles)

        header = create_scene_header(
            "CHAPTER 03  |  SINGULAR VALUE DECOMPOSITION",
            "Factoring Any Image Matrix: A = U Σ Vᵀ",
            accent_color=COLOR_DIAGONAL,
        )
        self.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.9)

        # --- 2:05 - 2:28 | Introduce A = U Sigma V^T as Separate Visual Blocks ---
        svd_eq = MathTex(r"A", r"=", r"U", r"\Sigma", r"V^\top", font_size=58)
        svd_eq[0].set_color(COLOR_ORIGINAL)
        svd_eq[2].set_color(COLOR_EIGENVECTOR)
        svd_eq[3].set_color(COLOR_DIAGONAL)
        svd_eq[4].set_color(COLOR_EIGENVECTOR)
        svd_eq.move_to(UP * 1.65)

        self.play(Write(svd_eq), run_time=1.8)

        # Matrix block geometry scaled to reflect m x n, m x r, r x r, r x n
        block_a = Rectangle(
            width=2.6, height=1.55, fill_color=COLOR_ORIGINAL, fill_opacity=0.14,
            stroke_color=COLOR_ORIGINAL, stroke_width=2.0
        )
        lbl_a = MathTex(f"A\\\\({m}\\times {n})", font_size=24, color=COLOR_ORIGINAL).move_to(block_a)

        eq_sign = MathTex("=", font_size=40, color=COLOR_WHITE)

        block_u = Rectangle(
            width=1.55, height=1.55, fill_color=COLOR_EIGENVECTOR, fill_opacity=0.16,
            stroke_color=COLOR_EIGENVECTOR, stroke_width=2.0
        )
        lbl_u = MathTex(f"U\\\\({m}\\times {r})", font_size=24, color=COLOR_EIGENVECTOR).move_to(block_u)

        block_s = Rectangle(
            width=1.55, height=1.55, fill_color=COLOR_DIAGONAL, fill_opacity=0.16,
            stroke_color=COLOR_DIAGONAL, stroke_width=2.2
        )
        diag_line = Line(
            block_s.get_corner(UP + LEFT) + RIGHT * 0.12 + DOWN * 0.12,
            block_s.get_corner(DOWN + RIGHT) + LEFT * 0.12 + UP * 0.12,
            stroke_color=COLOR_EIGENVALUE,
            stroke_width=3.5,
        )
        lbl_s = MathTex(r"\Sigma", font_size=28, color=COLOR_DIAGONAL).move_to(block_s.get_top() + DOWN * 0.35 + RIGHT * 0.35)

        block_vt = Rectangle(
            width=2.6, height=1.55, fill_color=COLOR_EIGENVECTOR, fill_opacity=0.16,
            stroke_color=COLOR_EIGENVECTOR, stroke_width=2.0
        )
        lbl_vt = MathTex(f"V^\\top\\\\({r}\\times {n})", font_size=24, color=COLOR_EIGENVECTOR).move_to(block_vt)

        g_a = VGroup(block_a, lbl_a)
        g_u = VGroup(block_u, lbl_u)
        g_s = VGroup(block_s, diag_line, lbl_s)
        g_vt = VGroup(block_vt, lbl_vt)

        blocks_row = VGroup(g_a, eq_sign, g_u, g_s, g_vt).arrange(RIGHT, buff=0.52)
        blocks_row.move_to(DOWN * 0.15)

        role_u = Text("Left singular vectors\n(vertical patterns)", font_size=14, color=COLOR_EIGENVECTOR)
        role_s = Text("Singular values\n(energy weights)", font_size=14, color=COLOR_DIAGONAL)
        role_vt = Text("Right singular vectors\n(horizontal patterns)", font_size=14, color=COLOR_EIGENVECTOR)

        role_u.next_to(g_u, DOWN, buff=0.25)
        role_s.next_to(g_s, DOWN, buff=0.25)
        role_vt.next_to(g_vt, DOWN, buff=0.25)

        self.play(FadeIn(g_a), FadeIn(eq_sign), run_time=1.2)
        self.play(FadeIn(g_u, shift=UP * 0.2), FadeIn(role_u), run_time=1.5)
        self.play(FadeIn(g_s, shift=UP * 0.2), FadeIn(role_s), run_time=1.5)
        self.play(FadeIn(g_vt, shift=UP * 0.2), FadeIn(role_vt), run_time=1.5)
        self.wait(3.8)

        # --- 2:28 - 2:45 | Link Eigenvalues of A^T A to Singular Values: lambda_i = sigma_i^2 ---
        self.play(
            FadeOut(blocks_row),
            FadeOut(role_u),
            FadeOut(role_s),
            FadeOut(role_vt),
            svd_eq.animate.scale(0.75).to_corner(UP + RIGHT, buff=0.48),
            run_time=1.1,
        )

        bridge_box = RoundedRectangle(
            width=11.4,
            height=4.0,
            corner_radius=0.16,
            fill_color=BG_PANEL,
            fill_opacity=0.92,
            stroke_color=COLOR_DIAGONAL,
            stroke_width=1.8,
        ).move_to(ORIGIN + DOWN * 0.25)

        derivation_1 = MathTex(
            r"A^\top A = (U \Sigma V^\top)^\top (U \Sigma V^\top) = V (\Sigma^\top \Sigma) V^\top = V \Lambda V^\top",
            font_size=34,
            color=COLOR_ORIGINAL,
        )
        derivation_2 = MathTex(
            r"\lambda_i = \sigma_i^2",
            font_size=54,
            color=COLOR_EIGENVALUE,
        )
        sigma_diag = MathTex(
            r"\Sigma = \operatorname{diag}(\sigma_1, \sigma_2, \dots, \sigma_r), \quad "
            r"\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0",
            font_size=30,
            color=COLOR_DIAGONAL,
        )
        verified_note = Text(
            f"Verified on our image matrix: max |λᵢ − σᵢ²| / max = {data.eigen_svd_max_rel_diff:.2e}",
            font_size=17,
            color=COLOR_SUBTEXT,
        )
        bridge_stack = VGroup(derivation_1, derivation_2, sigma_diag, verified_note).arrange(
            DOWN, buff=0.32
        )
        bridge_stack.move_to(bridge_box)

        self.play(FadeIn(bridge_box), Write(derivation_1), run_time=1.8)
        self.play(Write(derivation_2), run_time=1.3)
        self.play(FadeIn(sigma_diag, shift=UP * 0.15), FadeIn(verified_note), run_time=1.4)
        self.wait(2.5)
        self.play(FadeOut(bridge_box), FadeOut(bridge_stack), run_time=0.9)

        # --- 2:45 - 3:10 | Animate Real Singular Values in Descending Order ---
        num_bars = 50
        top_k = 20
        top_s = s_vals[:num_bars]
        max_s = float(top_s[0])

        axes = Axes(
            x_range=[0, num_bars + 2, 10],
            y_range=[0, 1.08, 0.25],
            x_length=9.8,
            y_length=4.0,
            axis_config={"color": COLOR_SUBTEXT, "stroke_width": 1.6},
            tips=False,
        ).move_to(ORIGIN + DOWN * 0.45)

        x_label = Text("Component index i (sorted descending)", font_size=18, color=COLOR_SUBTEXT)
        x_label.next_to(axes, DOWN, buff=0.25)
        y_label = MathTex(r"\sigma_i / \sigma_1", font_size=26, color=COLOR_CYAN)
        y_label.next_to(axes.y_axis, UP, buff=0.15)

        bars_kept = VGroup()
        bars_tail = VGroup()
        bar_w = (9.8 / (num_bars + 2)) * 0.74
        base_y = axes.c2p(0, 0)[1]

        for i, val in enumerate(top_s):
            norm_h = max(float(val) / max_s, 0.015)
            top_pt = axes.c2p(i + 1, norm_h)
            bot_pt = axes.c2p(i + 1, 0)
            h_px = max(top_pt[1] - bot_pt[1], 0.04)
            is_kept = (i + 1) <= top_k
            bar_color = COLOR_EIGENVALUE if (i + 1) <= 5 else (COLOR_CYAN if is_kept else COLOR_DISCARDED)
            bar_opacity = 0.92 if is_kept else 0.45
            rect = Rectangle(
                width=bar_w,
                height=h_px,
                fill_color=bar_color,
                fill_opacity=bar_opacity,
                stroke_width=0,
            )
            rect.move_to([top_pt[0], base_y + h_px / 2.0, 0])
            if is_kept:
                bars_kept.add(rect)
            else:
                bars_tail.add(rect)

        energy_20 = cum_energy[top_k - 1] * 100.0
        energy_5 = cum_energy[4] * 100.0

        spectrum_title = Text(
            f"Real Singular Value Spectrum (σ₁ = {s_vals[0]:,.0f}, σ₂₀ = {s_vals[19]:,.0f}, σ₅₀ = {s_vals[49]:,.0f})",
            font_size=20,
            color=COLOR_WHITE,
        ).next_to(axes, UP, buff=0.35)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), FadeIn(spectrum_title), run_time=1.3)
        self.play(
            FadeIn(bars_kept, lag_ratio=0.06),
            FadeIn(bars_tail, lag_ratio=0.04),
            run_time=2.5,
        )

        # Highlight dominant vs discarded components
        cutoff_x = axes.c2p(top_k + 0.5, 0)[0]
        cutoff_line = Line(
            [cutoff_x, axes.c2p(0, 0)[1], 0],
            [cutoff_x, axes.c2p(0, 1.0)[1], 0],
            stroke_color=COLOR_CYAN,
            stroke_width=2.2,
        )
        kept_callout = Text(
            f"Top 20 capture {energy_20:.1f}% of image energy",
            font_size=18,
            color=COLOR_CYAN,
            weight="BOLD",
        ).move_to(axes.c2p(36, 0.78))

        tail_callout = Text(
            "Tail components -> muted & discardable",
            font_size=17,
            color=COLOR_DISCARDED,
        ).move_to(axes.c2p(36, 0.24))

        energy5_callout = Text(
            f"Top 5: {energy_5:.1f}%",
            font_size=16,
            color=COLOR_EIGENVALUE,
        ).move_to(axes.c2p(10.5, 0.88))

        self.play(Create(cutoff_line), FadeIn(kept_callout), FadeIn(energy5_callout), run_time=1.6)
        self.play(FadeIn(tail_callout), run_time=1.2)
        self.wait(5.0)

        self.play(
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(spectrum_title),
            FadeOut(bars_kept),
            FadeOut(bars_tail),
            FadeOut(cutoff_line),
            FadeOut(kept_callout),
            FadeOut(energy5_callout),
            FadeOut(tail_callout),
            FadeOut(svd_eq),
            FadeOut(header),
            FadeOut(particles),
            run_time=1.0,
        )
