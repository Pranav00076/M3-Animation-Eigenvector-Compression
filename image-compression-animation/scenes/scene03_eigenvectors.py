"""Scene 3: Eigenvectors and Diagonalization (1:15 - 2:05)

Mathematically rigorous explanation:
- Real image matrix A (m x n) is rectangular, so it does NOT directly factor as P D P^-1.
- Instead, we form the symmetric positive semi-definite matrix A^T A (n x n).
- Demonstrates eigenvectors visually as special invariant directions under A^T A.
- Rotates into the orthonormal eigenvector basis V and reveals A^T A = V Lambda V^T.
- Highlights eigenvalues in purple and explains why changing basis simplifies structure.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from manim import (
    DOWN,
    DR,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Arrow,
    Axes,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    MathTex,
    NumberPlane,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    VGroup,
    Vector,
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
    COLOR_EIGENVALUE,
    COLOR_EIGENVECTOR,
    COLOR_ORIGINAL,
    COLOR_SUBTEXT,
    COLOR_WHITE,
    configure_scene_canvas,
    create_ambient_particles,
    create_scene_header,
    create_subtitle_caption,
    get_pipeline_data,
)


class Scene03Eigenvectors(Scene):
    """1:15 - 2:05 | Symmetric A^T A, eigenvectors, basis change, and Lambda."""

    def construct(self) -> None:
        configure_scene_canvas()
        data = get_pipeline_data()
        m, n = data.image_data.height, data.image_data.width
        top_evals = data.eigen_result.eigenvalues[:3]

        particles = create_ambient_particles(count=34, seed=303)
        self.add(particles)

        header = create_scene_header(
            "CHAPTER 02  |  SYMMETRY & BASIS",
            "Eigenvectors and Diagonalization",
            accent_color=COLOR_EIGENVECTOR,
        )
        self.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.9)

        # --- 1:15 - 1:27 | Why A^T A instead of P D P^-1 ---
        warn_box = RoundedRectangle(
            width=11.4,
            height=3.4,
            corner_radius=0.16,
            fill_color=BG_PANEL,
            fill_opacity=0.92,
            stroke_color=COLOR_EIGENVECTOR,
            stroke_width=1.8,
        ).move_to(ORIGIN + DOWN * 0.25)

        rect_eq = MathTex(
            r"A \in \mathbb{R}^{" + f"{m}" + r" \times " + f"{n}" + r"}"
            r"\quad\text{is rectangular }\Rightarrow\text{ no direct } P D P^{-1}",
            font_size=32,
            color=COLOR_ORIGINAL,
        )
        sym_eq = MathTex(
            r"A^\top A \in \mathbb{R}^{" + f"{n}" + r" \times " + f"{n}" + r"}"
            r"\quad\text{is symmetric: }\; (A^\top A)^\top = A^\top A",
            font_size=34,
            color=COLOR_CYAN,
        )
        sym_note = Text(
            "By the Spectral Theorem, its eigenvectors form an orthonormal basis.",
            font_size=21,
            color=COLOR_SUBTEXT,
        )
        intro_stack = VGroup(rect_eq, sym_eq, sym_note).arrange(DOWN, buff=0.35)
        intro_stack.move_to(warn_box)

        self.play(FadeIn(warn_box), Write(rect_eq), run_time=1.8)
        self.wait(1.6)
        self.play(Write(sym_eq), FadeIn(sym_note, shift=UP * 0.15), run_time=1.8)
        self.wait(2.4)
        self.play(FadeOut(warn_box), FadeOut(intro_stack), run_time=0.9)

        # --- 1:27 - 1:47 | Geometric Demo: Eigenvectors as Special Directions ---
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=6.4,
            y_length=4.8,
            background_line_style={
                "stroke_color": COLOR_EIGENVECTOR,
                "stroke_width": 1.0,
                "stroke_opacity": 0.25,
            },
            axis_config={
                "stroke_color": COLOR_SUBTEXT,
                "stroke_width": 1.4,
            },
        ).move_to(LEFT * 2.5 + DOWN * 0.3)

        # Symmetric 2x2 covariance illustration: rotated by 32 deg with eigenvalues 2.1 and 0.65
        theta = np.deg2rad(32.0)
        v1 = np.array([np.cos(theta), np.sin(theta)])
        v2 = np.array([-np.sin(theta), np.cos(theta)])
        lam1, lam2 = 2.0, 0.65
        mat_sym = lam1 * np.outer(v1, v1) + lam2 * np.outer(v2, v2)

        # Generic off-axis vector (rotates when transformed!)
        u_gen = np.array([0.95, -0.35])
        u_gen_t = mat_sym @ u_gen

        def plane_pt(vec2: np.ndarray) -> np.ndarray:
            return plane.c2p(float(vec2[0]), float(vec2[1]))

        vec_gen = Arrow(
            plane_pt(np.zeros(2)),
            plane_pt(u_gen),
            buff=0,
            color=COLOR_SUBTEXT,
            stroke_width=4,
        )
        vec_gen_label = MathTex(r"x", color=COLOR_SUBTEXT, font_size=26).next_to(
            vec_gen.get_end(), DR * 0.5, buff=0.08
        )

        # Eigenvector spans (dashed lines) and unit eigenvectors (electric blue)
        span1 = DashedLine(
            plane_pt(-2.5 * v1), plane_pt(2.5 * v1), color=COLOR_EIGENVECTOR, stroke_opacity=0.45
        )
        span2 = DashedLine(
            plane_pt(-2.2 * v2), plane_pt(2.2 * v2), color=COLOR_EIGENVECTOR, stroke_opacity=0.45
        )
        evec1 = Arrow(
            plane_pt(np.zeros(2)),
            plane_pt(1.15 * v1),
            buff=0,
            color=COLOR_EIGENVECTOR,
            stroke_width=5,
        )
        evec2 = Arrow(
            plane_pt(np.zeros(2)),
            plane_pt(1.15 * v2),
            buff=0,
            color=COLOR_EIGENVECTOR,
            stroke_width=5,
        )
        evec1_lbl = MathTex(r"v_1", color=COLOR_EIGENVECTOR, font_size=28).next_to(
            evec1.get_end(), UP * 0.5 + RIGHT * 0.3, buff=0.08
        )
        evec2_lbl = MathTex(r"v_2", color=COLOR_EIGENVECTOR, font_size=28).next_to(
            evec2.get_end(), UP * 0.5 + LEFT * 0.3, buff=0.08
        )

        # Right-side explanation panel
        panel = RoundedRectangle(
            width=5.1,
            height=4.8,
            corner_radius=0.14,
            fill_color=BG_PANEL,
            fill_opacity=0.9,
            stroke_color=COLOR_EIGENVECTOR,
            stroke_width=1.5,
        ).move_to(RIGHT * 3.7 + DOWN * 0.3)

        p_title = Text("Special Directions", font_size=24, color=COLOR_WHITE, weight="BOLD")
        p_eq = MathTex(
            r"(A^\top A)\, v_i = \lambda_i\, v_i",
            font_size=34,
            color=COLOR_ORIGINAL,
        )
        p_eq[0][5:7].set_color(COLOR_EIGENVECTOR)
        p_eq[0][8:10].set_color(COLOR_EIGENVALUE)
        p_eq[0][10:12].set_color(COLOR_EIGENVECTOR)

        p_desc1 = Text(
            "• Most vectors rotate off their line.",
            font_size=18,
            color=COLOR_SUBTEXT,
        )
        p_desc2 = Text(
            "• Eigenvectors only stretch by λᵢ.",
            font_size=18,
            color=COLOR_EIGENVECTOR,
        )
        p_desc3 = Text(
            "• Orthogonal: v₁ᵀ v₂ = 0",
            font_size=18,
            color=COLOR_CYAN,
        )
        panel_content = VGroup(p_title, p_eq, p_desc1, p_desc2, p_desc3).arrange(
            DOWN, aligned_edge=LEFT, buff=0.28
        )
        panel_content.move_to(panel)

        self.play(Create(plane), FadeIn(panel), FadeIn(panel_content), run_time=1.4)
        self.play(
            Create(span1),
            Create(span2),
            Create(evec1),
            Create(evec2),
            FadeIn(evec1_lbl),
            FadeIn(evec2_lbl),
            Create(vec_gen),
            FadeIn(vec_gen_label),
            run_time=1.6,
        )
        self.wait(1.2)

        # Animate action of A^T A: generic vector rotates off its line; eigenvectors stretch purely along span
        vec_gen_transformed = Arrow(
            plane_pt(np.zeros(2)),
            plane_pt(u_gen_t),
            buff=0,
            color="#FBBF24",
            stroke_width=4,
        )
        vec_gen_t_lbl = MathTex(r"(A^\top A)x", color="#FBBF24", font_size=24).next_to(
            vec_gen_transformed.get_end(), RIGHT, buff=0.1
        )
        evec1_scaled = Arrow(
            plane_pt(np.zeros(2)),
            plane_pt(lam1 * 1.15 * v1),
            buff=0,
            color=COLOR_EIGENVECTOR,
            stroke_width=5.5,
        )
        evec2_scaled = Arrow(
            plane_pt(np.zeros(2)),
            plane_pt(lam2 * 1.15 * v2),
            buff=0,
            color=COLOR_EIGENVECTOR,
            stroke_width=5.5,
        )
        lam1_tag = MathTex(r"\lambda_1 v_1", color=COLOR_EIGENVALUE, font_size=26).next_to(
            evec1_scaled.get_end(), UP * 0.4, buff=0.08
        )
        lam2_tag = MathTex(r"\lambda_2 v_2", color=COLOR_EIGENVALUE, font_size=26).next_to(
            evec2_scaled.get_end(), LEFT * 0.5, buff=0.08
        )

        self.play(
            Transform(vec_gen, vec_gen_transformed),
            Transform(vec_gen_label, vec_gen_t_lbl),
            Transform(evec1, evec1_scaled),
            Transform(evec2, evec2_scaled),
            Transform(evec1_lbl, lam1_tag),
            Transform(evec2_lbl, lam2_tag),
            run_time=2.4,
            rate_func=smooth,
        )
        self.wait(2.4)

        # --- 1:47 - 2:05 | Basis Change & Diagonal Matrix Lambda ---
        # Rotate the visual basis cleanly to align eigenvectors with horizontal/vertical axes
        geo_group = VGroup(
            plane,
            span1,
            span2,
            evec1,
            evec2,
            evec1_lbl,
            evec2_lbl,
            vec_gen,
            vec_gen_label,
        )
        self.play(
            FadeOut(vec_gen),
            FadeOut(vec_gen_label),
            FadeOut(panel),
            FadeOut(panel_content),
            run_time=0.9,
        )
        self.play(
            geo_group.animate.rotate(-theta, about_point=plane.get_center()).scale(0.8).to_edge(LEFT, buff=0.7),
            run_time=1.8,
            rate_func=smooth,
        )

        # Show A^T A = V Lambda V^T and highlight eigenvalues in purple + diagonal matrix in cyan
        eq_decomp = MathTex(
            r"A^\top A", r"=", r"V", r"\Lambda", r"V^\top",
            font_size=48,
        )
        eq_decomp[0].set_color(COLOR_ORIGINAL)
        eq_decomp[1].set_color(COLOR_WHITE)
        eq_decomp[2].set_color(COLOR_EIGENVECTOR)
        eq_decomp[3].set_color(COLOR_DIAGONAL)
        eq_decomp[4].set_color(COLOR_EIGENVECTOR)

        lambda_matrix = MathTex(
            r"\Lambda = \begin{bmatrix}"
            r"\lambda_1 & 0 & \cdots & 0 \\"
            r"0 & \lambda_2 & \cdots & 0 \\"
            r"\vdots & \vdots & \ddots & \vdots \\"
            r"0 & 0 & \cdots & \lambda_n"
            r"\end{bmatrix}",
            font_size=34,
            color=COLOR_DIAGONAL,
        )

        real_vals_text = Text(
            f"Top eigenvalues: λ₁ = {top_evals[0]:,.0f},  λ₂ = {top_evals[1]:,.0f}\nλ₃ = {top_evals[2]:,.0f}",
            font_size=16,
            color=COLOR_EIGENVALUE,
            line_spacing=1.2,
        )
        basis_insight = Text(
            "Changing into the V-basis decouples every\ndirection into a clean diagonal matrix.",
            font_size=18,
            color=COLOR_WHITE,
            line_spacing=1.2,
        )

        right_stack = VGroup(eq_decomp, lambda_matrix, real_vals_text, basis_insight).arrange(
            DOWN, buff=0.32
        )
        right_stack.move_to(RIGHT * 2.55 + DOWN * 0.25)

        self.play(Write(eq_decomp), run_time=1.8)
        self.play(FadeIn(lambda_matrix, shift=UP * 0.2), run_time=1.6)
        self.play(FadeIn(real_vals_text), FadeIn(basis_insight), run_time=1.6)
        self.wait(5.0)

        self.play(
            FadeOut(geo_group),
            FadeOut(right_stack),
            FadeOut(header),
            FadeOut(particles),
            run_time=1.1,
        )
