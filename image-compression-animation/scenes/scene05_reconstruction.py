"""Scene 5: Low-Rank Image Reconstruction (3:10 - 4:20)

Uses actual NumPy SVD reconstructions for k = 1, 5, 10, 20, and full rank.
Shows side-by-side comparison with the original grayscale photograph, exact
stored values k(m+n+1) vs m*n, and live computed Frobenius error and PSNR.
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
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
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
    COLOR_DISCARDED,
    COLOR_EIGENVALUE,
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


class Scene05Reconstruction(Scene):
    """3:10 - 4:20 | Progressive rank-k SVD reconstruction & error comparison."""

    def construct(self) -> None:
        configure_scene_canvas()
        data = get_pipeline_data()
        m, n = data.image_data.height, data.image_data.width
        orig_gray_path = data.export_dir / "original_gray.png"

        particles = create_ambient_particles(count=34, seed=505)
        self.add(particles)

        header = create_scene_header(
            "CHAPTER 04  |  LOW-RANK APPROXIMATION",
            "Reconstructing the Image from Top Singular Components",
            accent_color=COLOR_RECON,
        )
        self.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.9)

        # --- 3:10 - 3:24 | Truncated Formula & Storage Accounting ---
        trunc_eq = MathTex(
            r"A_k = U_k \Sigma_k V_k^\top = \sum_{i=1}^{k} \sigma_i u_i v_i^\top",
            font_size=44,
            color=COLOR_ORIGINAL,
        )
        storage_eq = MathTex(
            r"\text{Original: } m \times n \quad\longrightarrow\quad "
            r"\text{Truncated: } k(m + n + 1)",
            font_size=34,
            color=COLOR_CYAN,
        )
        formula_note = Text(
            "Changing basis makes the structure simple; discarding small σᵢ saves storage.",
            font_size=20,
            color=COLOR_SUBTEXT,
        )
        formula_box = RoundedRectangle(
            width=11.4,
            height=3.4,
            corner_radius=0.16,
            fill_color=BG_PANEL,
            fill_opacity=0.92,
            stroke_color=COLOR_RECON,
            stroke_width=1.8,
        ).move_to(ORIGIN + DOWN * 0.25)
        formula_stack = VGroup(trunc_eq, storage_eq, formula_note).arrange(DOWN, buff=0.35)
        formula_stack.move_to(formula_box)

        self.play(FadeIn(formula_box), Write(trunc_eq), run_time=1.8)
        self.play(Write(storage_eq), FadeIn(formula_note), run_time=1.6)
        self.wait(2.2)
        self.play(FadeOut(formula_box), FadeOut(formula_stack), run_time=0.9)

        # --- 3:24 - 4:20 | Side-by-Side Progressive Reconstruction ---
        orig_img = ImageMobject(str(orig_gray_path)).set_width(5.6).move_to(LEFT * 3.35 + UP * 0.1)
        orig_frame = create_glow_frame(orig_img, color=COLOR_ORIGINAL, buff=0.05)
        orig_label = Text(
            f"Original Matrix A  ({m}x{n} = {m*n:,} values)",
            font_size=18,
            color=COLOR_ORIGINAL,
            weight="BOLD",
        ).next_to(orig_img, UP, buff=0.22)

        self.play(FadeIn(orig_img), Create(orig_frame), FadeIn(orig_label), run_time=1.3)

        # Bottom telemetry bar frame
        metrics_bg = RoundedRectangle(
            width=12.4,
            height=1.55,
            corner_radius=0.14,
            fill_color=BG_PANEL,
            fill_opacity=0.92,
            stroke_color=COLOR_CYAN,
            stroke_width=1.5,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(metrics_bg), run_time=0.7)

        current_recon_img = None
        current_recon_frame = None
        current_recon_label = None
        current_metrics_group = None

        for idx, rec in enumerate(data.reconstructions):
            is_full = rec.rank_k == data.svd_result.full_rank
            rank_title = (
                f"Full Rank A  (k = {rec.rank_k})"
                if is_full
                else f"Rank-{rec.rank_k} Approximation A_{rec.rank_k}"
            )
            recon_img = (
                ImageMobject(str(rec.image_path))
                .set_width(5.6)
                .move_to(RIGHT * 3.35 + UP * 0.1)
            )
            recon_frame = create_glow_frame(recon_img, color=COLOR_RECON, buff=0.05)
            recon_label = Text(
                rank_title,
                font_size=18,
                color=COLOR_RECON,
                weight="BOLD",
            ).next_to(recon_img, UP, buff=0.22)

            # Build 4 clear telemetry columns inside metrics_bg
            col1 = VGroup(
                Text("STORED VALUES", font_size=13, color=COLOR_SUBTEXT),
                Text(
                    f"{rec.stored_values:,} / {rec.original_values:,}",
                    font_size=21,
                    color=COLOR_WHITE,
                    weight="BOLD",
                ),
            ).arrange(DOWN, buff=0.1)

            col2 = VGroup(
                Text("STORAGE SAVED", font_size=13, color=COLOR_SUBTEXT),
                Text(
                    f"{rec.space_saved_percent:.1f}%",
                    font_size=21,
                    color=COLOR_CYAN if rec.space_saved_percent > 0 else COLOR_DISCARDED,
                    weight="BOLD",
                ),
            ).arrange(DOWN, buff=0.1)

            col3 = VGroup(
                Text("RELATIVE ERROR ||A - A_k|| / ||A||", font_size=13, color=COLOR_SUBTEXT),
                Text(
                    f"{rec.frobenius_relative_error * 100:.2f}%",
                    font_size=21,
                    color=COLOR_EIGENVALUE,
                    weight="BOLD",
                ),
            ).arrange(DOWN, buff=0.1)

            col4 = VGroup(
                Text("ENERGY RETAINED / PSNR", font_size=13, color=COLOR_SUBTEXT),
                Text(
                    f"{rec.energy_retained_percent:.1f}%  |  {rec.psnr_db:.1f} dB",
                    font_size=21,
                    color=COLOR_RECON,
                    weight="BOLD",
                ),
            ).arrange(DOWN, buff=0.1)

            metrics_row = VGroup(col1, col2, col3, col4).arrange(RIGHT, buff=0.75)
            metrics_row.move_to(metrics_bg)

            if current_recon_img is None:
                self.play(
                    FadeIn(recon_img),
                    Create(recon_frame),
                    FadeIn(recon_label),
                    FadeIn(metrics_row),
                    run_time=1.4,
                )
            else:
                self.play(
                    FadeIn(recon_img),
                    FadeOut(current_recon_img),
                    Transform(current_recon_frame, recon_frame),
                    Transform(current_recon_label, recon_label),
                    Transform(current_metrics_group, metrics_row),
                    run_time=1.35,
                )
                # Keep references clean after Transform
                if current_recon_frame is not None:
                    self.remove(current_recon_frame)
                if current_recon_label is not None:
                    self.remove(current_recon_label)
                if current_metrics_group is not None:
                    self.remove(current_metrics_group)
                self.add(recon_frame, recon_label, metrics_row)

            current_recon_img = recon_img
            current_recon_frame = recon_frame
            current_recon_label = recon_label
            current_metrics_group = metrics_row

            wait_sec = 4.8 if rec.rank_k in (1, 20) else 4.2
            self.wait(wait_sec)

        self.play(
            FadeOut(orig_img),
            FadeOut(orig_frame),
            FadeOut(orig_label),
            FadeOut(current_recon_img),
            FadeOut(current_recon_frame),
            FadeOut(current_recon_label),
            FadeOut(current_metrics_group),
            FadeOut(metrics_bg),
            FadeOut(header),
            FadeOut(particles),
            run_time=1.0,
        )
