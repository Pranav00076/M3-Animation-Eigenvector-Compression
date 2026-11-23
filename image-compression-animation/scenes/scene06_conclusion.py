"""Scene 6: Conclusion (4:20 - 5:00)

Side-by-side comparison of original vs rank-20 compressed image, exact storage
reduction summary, concise recap of eigenvectors, eigenvalues, SVD, and low-rank
approximation, ending with 'Less data. Nearly the same picture.' and credits.
"""

from __future__ import annotations

from pathlib import Path
import sys

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    ImageMobject,
    Line,
    MathTex,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    Write,
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
    COLOR_RECON,
    COLOR_SUBTEXT,
    COLOR_WHITE,
    configure_scene_canvas,
    create_ambient_particles,
    create_glow_frame,
    create_scene_header,
    get_pipeline_data,
)


class Scene06Conclusion(Scene):
    """4:20 - 5:00 | Side-by-side payoff, conceptual summary, and final credits."""

    def construct(self) -> None:
        configure_scene_canvas()
        data = get_pipeline_data()
        orig_gray_path = data.export_dir / "original_gray.png"

        # Pick k=20 reconstruction (or closest available below full rank)
        rec_20 = next(
            (r for r in data.reconstructions if r.rank_k == 20),
            data.reconstructions[-2],
        )

        particles = create_ambient_particles(count=42, seed=606)
        self.add(particles)

        header = create_scene_header(
            "CHAPTER 05  |  THE PAYOFF",
            "Compression Through Linear Algebra",
            accent_color=COLOR_CYAN,
        )
        self.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.9)

        # --- 4:20 - 4:36 | Side-by-Side Final Comparison & Storage Reduction ---
        orig_img = ImageMobject(str(orig_gray_path)).set_width(5.3).move_to(LEFT * 3.3 + UP * 0.25)
        orig_frame = create_glow_frame(orig_img, color=COLOR_ORIGINAL, buff=0.05)
        orig_caption = Text(
            f"Original: {rec_20.original_values:,} numbers",
            font_size=20,
            color=COLOR_ORIGINAL,
            weight="BOLD",
        ).next_to(orig_img, DOWN, buff=0.25)

        comp_img = ImageMobject(str(rec_20.image_path)).set_width(5.3).move_to(RIGHT * 3.3 + UP * 0.25)
        comp_frame = create_glow_frame(comp_img, color=COLOR_RECON, buff=0.05)
        comp_caption = Text(
            f"Rank-{rec_20.rank_k} SVD: {rec_20.stored_values:,} numbers",
            font_size=20,
            color=COLOR_RECON,
            weight="BOLD",
        ).next_to(comp_img, DOWN, buff=0.25)

        reduction_text = Text(
            f"{rec_20.space_saved_percent:.1f}% Storage Reduction   |   "
            f"{rec_20.energy_retained_percent:.1f}% Energy Retained   |   "
            f"PSNR {rec_20.psnr_db:.1f} dB",
            font_size=20,
            color=COLOR_CYAN,
            weight="BOLD",
        )
        reduction_banner = RoundedRectangle(
            width=max(reduction_text.width + 1.0, 11.2),
            height=0.95,
            corner_radius=0.14,
            fill_color=BG_PANEL,
            fill_opacity=0.92,
            stroke_color=COLOR_CYAN,
            stroke_width=1.8,
        ).to_edge(DOWN, buff=0.4)
        reduction_text.move_to(reduction_banner)

        self.play(
            FadeIn(orig_img),
            Create(orig_frame),
            FadeIn(orig_caption),
            FadeIn(comp_img),
            Create(comp_frame),
            FadeIn(comp_caption),
            run_time=2.0,
        )
        self.play(FadeIn(reduction_banner), FadeIn(reduction_text), run_time=1.3)
        self.wait(4.5)

        self.play(
            FadeOut(orig_img),
            FadeOut(orig_frame),
            FadeOut(orig_caption),
            FadeOut(comp_img),
            FadeOut(comp_frame),
            FadeOut(comp_caption),
            FadeOut(reduction_banner),
            FadeOut(reduction_text),
            run_time=1.0,
        )

        # --- 4:36 - 4:48 | Four-Card Conceptual Summary ---
        cards_spec = [
            ("Eigenvectors", r"A^\top A = V \Lambda V^\top", "Orthonormal basis directions", COLOR_EIGENVECTOR),
            ("Eigenvalues", r"\lambda_i = \sigma_i^2", "Measure directional energy", COLOR_EIGENVALUE),
            ("SVD", r"A = U \Sigma V^\top", "Exact basis change for any A", COLOR_DIAGONAL),
            ("Low-Rank Truncation", r"A_k = U_k \Sigma_k V_k^\top", "Discard small σᵢ to save data", COLOR_RECON),
        ]

        cards = VGroup()
        for title, eq_tex, desc, col in cards_spec:
            box = RoundedRectangle(
                width=5.6,
                height=2.15,
                corner_radius=0.14,
                fill_color=BG_PANEL,
                fill_opacity=0.92,
                stroke_color=col,
                stroke_width=1.8,
            )
            t = Text(title, font_size=21, color=col, weight="BOLD")
            eq = MathTex(eq_tex, font_size=30, color=COLOR_WHITE)
            d = Text(desc, font_size=17, color=COLOR_SUBTEXT)
            inner = VGroup(t, eq, d).arrange(DOWN, buff=0.18).move_to(box)
            cards.add(VGroup(box, inner))

        cards.arrange_in_grid(rows=2, cols=2, buff=0.45).move_to(ORIGIN + DOWN * 0.25)

        self.play(FadeIn(cards, lag_ratio=0.18), run_time=2.4)
        self.wait(3.5)
        self.play(FadeOut(cards), FadeOut(header), run_time=1.0)

        # --- 4:48 - 5:00 | Final Statement & Credits ---
        closing_line = Text(
            "Less data. Nearly the same picture.",
            font_size=44,
            color=COLOR_WHITE,
            weight="BOLD",
        )
        closing_rule = Line(LEFT * 4.2, RIGHT * 4.2, stroke_color=COLOR_CYAN, stroke_width=2.2)
        final_title = Text(
            "The Mathematics Behind Image Compression",
            font_size=28,
            color=COLOR_CYAN,
            weight="SEMIBOLD",
        )
        credits = Text(
            "Built with Manim Community Edition · NumPy · SciPy · Pillow",
            font_size=18,
            color=COLOR_SUBTEXT,
        )
        finale = VGroup(closing_line, closing_rule, final_title, credits).arrange(
            DOWN, buff=0.35
        ).move_to(ORIGIN)

        self.play(Write(closing_line), run_time=1.8)
        self.play(Create(closing_rule), FadeIn(final_title), FadeIn(credits), run_time=1.5)
        self.wait(3.5)
        self.play(FadeOut(finale), FadeOut(particles), run_time=1.2)
