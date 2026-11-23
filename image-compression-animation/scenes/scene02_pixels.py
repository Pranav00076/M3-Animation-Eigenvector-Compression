"""Scene 2: Images as Matrices (0:35 - 1:15)

Transforms the color photograph into grayscale, zooms into its pixel grid,
maps pixel brightness to integers [0, 255] using real computed patch values,
constructs a small matrix, and transitions to the full image matrix A in R^(m x n).
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
    Create,
    FadeIn,
    FadeOut,
    ImageMobject,
    IntegerMatrix,
    Line,
    MathTex,
    MovingCameraScene,
    Rectangle,
    RoundedRectangle,
    Square,
    Text,
    Transform,
    UR,
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
    COLOR_EIGENVALUE,
    COLOR_ORIGINAL,
    COLOR_SUBTEXT,
    COLOR_WHITE,
    configure_scene_canvas,
    create_ambient_particles,
    create_glow_frame,
    create_metric_badge,
    create_scene_header,
    create_subtitle_caption,
    get_pipeline_data,
)


class Scene02Pixels(MovingCameraScene):
    """0:35 - 1:15 | From color photograph to grayscale matrix A."""

    def construct(self) -> None:
        configure_scene_canvas()
        data = get_pipeline_data()
        color_img_path = data.export_dir / "original_color.png"
        gray_img_path = data.export_dir / "original_gray.png"

        m, n = data.image_data.height, data.image_data.width
        patch = data.sample_patch  # 6x6 integer numpy array from real image

        particles = create_ambient_particles(count=32, seed=202)
        self.add(particles)

        header = create_scene_header(
            "CHAPTER 01  |  REPRESENTATION",
            "Images as Matrices",
            accent_color=COLOR_CYAN,
        )
        self.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.9)

        # --- 0:35 - 0:46 | Color to Grayscale Transformation ---
        color_photo = ImageMobject(str(color_img_path)).set_width(7.8).move_to(ORIGIN + DOWN * 0.25)
        gray_photo = ImageMobject(str(gray_img_path)).set_width(7.8).move_to(ORIGIN + DOWN * 0.25)
        frame = create_glow_frame(color_photo, color=COLOR_CYAN, buff=0.05)

        caption_rgb = create_subtitle_caption(
            "First, collapse 3 color channels into a single luminance channel.",
            color=COLOR_ORIGINAL,
        )

        self.play(FadeIn(color_photo), Create(frame), FadeIn(caption_rgb), run_time=1.3)
        self.wait(1.6)

        # Smooth sweep transition from RGB to grayscale
        self.play(
            FadeIn(gray_photo),
            FadeOut(color_photo),
            run_time=2.0,
        )
        gray_badge = create_metric_badge(
            "GRAYSCALE RANGE",
            "0 (Black)  to  255 (White)",
            accent_color=COLOR_CYAN,
            width=4.2,
        ).to_corner(UR, buff=0.45)
        self.play(FadeIn(gray_badge, shift=LEFT * 0.2), run_time=0.9)
        self.wait(1.8)

        # --- 0:46 - 1:02 | Zoom Into Pixel Grid & Assign Numbers ---
        self.play(FadeOut(caption_rgb), FadeOut(gray_badge), run_time=0.7)

        # Build an authentic 6x6 grayscale tile grid from the real image patch
        cell_size = 0.62
        grid_group = VGroup()
        number_group = VGroup()
        rows, cols = patch.shape

        for r in range(rows):
            for c in range(cols):
                val = int(patch[r, c])
                intensity = val / 255.0
                # Map brightness to grayscale fill
                hex_v = f"#{val:02x}{val:02x}{val:02x}"
                cell = Square(
                    side_length=cell_size,
                    fill_color=hex_v,
                    fill_opacity=1.0,
                    stroke_color=COLOR_CYAN,
                    stroke_width=1.0,
                    stroke_opacity=0.65,
                )
                cell.move_to(
                    [
                        (c - (cols - 1) / 2.0) * cell_size,
                        ((rows - 1) / 2.0 - r) * cell_size,
                        0,
                    ]
                )
                txt_color = "#050914" if intensity > 0.55 else "#E6F9FF"
                num = Text(str(val), font_size=15, color=txt_color, weight="BOLD")
                num.move_to(cell.get_center())
                grid_group.add(cell)
                number_group.add(num)

        patch_center = ORIGIN + LEFT * 2.7 + DOWN * 0.25
        grid_group.move_to(patch_center)
        number_group.move_to(patch_center)

        # Highlight patch location on grayscale image, then transition to numeric grid
        spot = Square(
            side_length=0.65,
            stroke_color=COLOR_CYAN,
            stroke_width=2.8,
        ).move_to(gray_photo.get_center() + LEFT * 0.8 + DOWN * 0.1)

        self.play(Create(spot), run_time=0.8)
        self.play(
            gray_photo.animate.set_width(3.4).to_edge(LEFT, buff=0.6).shift(DOWN * 0.3),
            FadeOut(frame),
            FadeOut(spot),
            run_time=1.4,
            rate_func=smooth,
        )

        # Center the pixel grid on the right half first, then construct matrix
        grid_group.move_to(ORIGIN + RIGHT * 1.4 + DOWN * 0.25)
        number_group.move_to(grid_group.get_center())

        grid_label = Text("6 x 6 Pixel Sample", font_size=18, color=COLOR_CYAN)
        grid_label.next_to(grid_group, UP, buff=0.22)

        self.play(FadeIn(grid_group, scale=0.85), FadeIn(grid_label), run_time=1.3)
        self.play(FadeIn(number_group, lag_ratio=0.02), run_time=1.6)
        self.wait(1.8)

        # --- 1:02 - 1:15 | Construct Small Matrix & Expand to Full Image Matrix A ---
        self.play(
            FadeOut(gray_photo),
            FadeOut(grid_label),
            grid_group.animate.move_to(LEFT * 3.5 + DOWN * 0.25),
            number_group.animate.move_to(LEFT * 3.5 + DOWN * 0.25),
            run_time=1.3,
        )

        small_matrix = IntegerMatrix(
            patch,
            left_bracket="[",
            right_bracket="]",
            h_buff=1.25,
            v_buff=0.62,
        ).scale(0.56)
        small_matrix.set_color(COLOR_ORIGINAL)
        small_matrix.get_brackets().set_color(COLOR_CYAN)
        small_matrix.move_to(RIGHT * 2.75 + DOWN * 0.25)

        arrow = MathTex(r"\Longrightarrow", color=COLOR_CYAN, font_size=46)
        arrow.move_to(LEFT * 0.75 + DOWN * 0.25)

        patch_title = MathTex(
            r"A_{\text{patch}} \in \mathbb{R}^{6 \times 6}",
            color=COLOR_ORIGINAL,
            font_size=28,
        ).next_to(small_matrix, UP, buff=0.25)

        self.play(Write(arrow), FadeIn(small_matrix), FadeIn(patch_title), run_time=1.8)
        self.wait(3.8)

        # Transition from small 6x6 patch to full m x n image matrix A
        full_matrix_eq = MathTex(
            r"A \in \mathbb{R}^{" + f"{m}" + r" \times " + f"{n}" + r"}",
            font_size=48,
            color=COLOR_ORIGINAL,
        )
        full_count = Text(
            f"{m} rows x {n} columns  =  {m * n:,} numbers",
            font_size=24,
            color=COLOR_CYAN,
            weight="BOLD",
        )
        full_note = Text(
            "Every pixel brightness is now an entry A[i, j] in a rectangular matrix.",
            font_size=20,
            color=COLOR_SUBTEXT,
        )
        summary_box = RoundedRectangle(
            width=10.6,
            height=3.1,
            corner_radius=0.16,
            fill_color=BG_PANEL,
            fill_opacity=0.9,
            stroke_color=COLOR_CYAN,
            stroke_width=1.8,
        )
        summary_text = VGroup(full_matrix_eq, full_count, full_note).arrange(DOWN, buff=0.28)
        summary_text.move_to(summary_box)
        summary_panel = VGroup(summary_box, summary_text).move_to(ORIGIN + DOWN * 0.25)

        self.play(
            FadeOut(grid_group),
            FadeOut(number_group),
            FadeOut(arrow),
            FadeOut(patch_title),
            Transform(small_matrix, summary_panel),
            run_time=2.0,
            rate_func=smooth,
        )
        self.wait(4.5)
        self.play(FadeOut(small_matrix), FadeOut(header), FadeOut(particles), run_time=1.1)
