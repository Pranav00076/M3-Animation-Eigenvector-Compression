"""Scene 1: The Mystery (0:00 - 0:35)

Introduces the high-resolution photograph, zooms in smoothly with MovingCameraScene
to reveal individual pixels, displays illustrative file/raw storage size, and poses
the central mathematical question.
"""

from __future__ import annotations

from pathlib import Path
import sys

from manim import (
    DOWN,
    DR,
    FadeIn,
    FadeOut,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Create,
    ImageMobject,
    Line,
    MovingCameraScene,
    Rectangle,
    RoundedRectangle,
    Square,
    Text,
    Transform,
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
    COLOR_ACCENT_GOLD,
    COLOR_CYAN,
    COLOR_EIGENVALUE,
    COLOR_EIGENVECTOR,
    COLOR_ORIGINAL,
    COLOR_SUBTEXT,
    COLOR_WHITE,
    configure_scene_canvas,
    create_ambient_particles,
    create_glow_frame,
    create_metric_badge,
    create_subtitle_caption,
    get_pipeline_data,
)


class Scene01Intro(MovingCameraScene):
    """0:00 - 0:35 | Documentary opening & pixel zoom mystery."""

    def construct(self) -> None:
        configure_scene_canvas()
        data = get_pipeline_data()
        color_img_path = data.export_dir / "original_color.png"
        zoom_color_path = data.export_dir / "zoom_pixels_color.png"

        particles = create_ambient_particles(count=42, seed=101)
        self.add(particles)

        # --- 0:00 - 0:08 | Cinematic Title Sequence ---
        kicker = Text(
            "LINEAR ALGEBRA IN PRACTICE",
            font_size=18,
            color=COLOR_CYAN,
            weight="BOLD",
        )
        main_title = Text(
            "The Mathematics Behind\nImage Compression",
            font_size=48,
            color=COLOR_WHITE,
            weight="BOLD",
            line_spacing=1.15,
        )
        subtitle = Text(
            "How Eigenvectors and Linear Algebra Help Reduce Image Data",
            font_size=22,
            color=COLOR_SUBTEXT,
        )
        accent_line = Line(LEFT * 3.8, RIGHT * 3.8, stroke_color=COLOR_CYAN, stroke_width=2.2)

        title_group = VGroup(kicker, main_title, accent_line, subtitle).arrange(
            DOWN, buff=0.28
        )
        title_group.move_to(ORIGIN)

        self.play(FadeIn(particles, run_time=1.0))
        self.play(FadeIn(kicker, shift=UP * 0.2), run_time=0.8)
        self.play(Write(main_title), run_time=1.8)
        self.play(Create(accent_line), FadeIn(subtitle, shift=UP * 0.15), run_time=1.2)
        self.wait(2.2)
        self.play(FadeOut(title_group, shift=UP * 0.3), run_time=1.0)

        # --- 0:08 - 0:18 | Introduce High-Resolution Photograph & File Size ---
        photo = ImageMobject(str(color_img_path))
        photo.set_width(8.6)
        photo.move_to(ORIGIN + UP * 0.52)
        frame = create_glow_frame(photo, color=COLOR_CYAN, buff=0.06)

        m, n = data.image_data.height, data.image_data.width
        file_kb = data.image_data.original_file_bytes / 1024.0

        size_badge = create_metric_badge(
            "MATRIX DIMENSIONS",
            f"{m} x {n} px",
            accent_color=COLOR_CYAN,
            width=2.6,
            height=0.78,
        )
        bytes_badge = create_metric_badge(
            "ILLUSTRATIVE FILE SIZE",
            f"{file_kb:.0f} KB ({m*n*3:,} RGB vals)",
            accent_color=COLOR_EIGENVALUE,
            width=4.0,
            height=0.78,
        )
        badges = VGroup(size_badge, bytes_badge).arrange(RIGHT, buff=0.25)
        badges.to_corner(DR, buff=0.35)

        caption1 = create_subtitle_caption(
            "Hundreds of thousands of hidden numbers.",
            color=COLOR_ORIGINAL,
        )
        caption1.scale(0.84)
        caption1.to_edge(DOWN, buff=0.40).to_edge(LEFT, buff=0.45)

        self.play(FadeIn(photo, scale=0.96), Create(frame), run_time=1.8)
        self.play(FadeIn(badges, shift=UP * 0.2), FadeIn(caption1), run_time=1.3)
        self.wait(4.0)

        # --- 0:18 - 0:28 | Camera Zoom Into Photograph & Reveal Pixels ---
        zoom_target_point = photo.get_center() + LEFT * 0.85 + DOWN * 0.12
        reticle = Square(
            side_length=1.1,
            stroke_color=COLOR_ACCENT_GOLD,
            stroke_width=2.5,
        ).move_to(zoom_target_point)

        self.play(
            FadeOut(badges),
            FadeOut(caption1),
            Create(reticle),
            run_time=0.9,
        )

        # Smooth MovingCamera zoom toward the observatory glass
        self.play(
            self.camera.frame.animate.set(width=3.2).move_to(zoom_target_point),
            reticle.animate.set_stroke(opacity=0.3),
            run_time=2.8,
            rate_func=smooth,
        )

        # Crossfade smoothly to crisp nearest-neighbor pixel reveal
        pixel_crop = ImageMobject(str(zoom_color_path))
        pixel_crop.set_width(3.0)
        pixel_crop.move_to(zoom_target_point)

        grid_lines = VGroup()
        rows, cols = 12, 20
        w_step = pixel_crop.width / cols
        h_step = pixel_crop.height / rows
        left_x = pixel_crop.get_left()[0]
        bottom_y = pixel_crop.get_bottom()[1]
        for c in range(cols + 1):
            x = left_x + c * w_step
            grid_lines.add(
                Line(
                    [x, bottom_y, 0],
                    [x, bottom_y + pixel_crop.height, 0],
                    stroke_color=COLOR_CYAN,
                    stroke_width=0.6,
                    stroke_opacity=0.35,
                )
            )
        for r in range(rows + 1):
            y = bottom_y + r * h_step
            grid_lines.add(
                Line(
                    [left_x, y, 0],
                    [left_x + pixel_crop.width, y, 0],
                    stroke_color=COLOR_CYAN,
                    stroke_width=0.6,
                    stroke_opacity=0.35,
                )
            )

        self.play(
            FadeIn(pixel_crop),
            FadeOut(photo),
            FadeOut(frame),
            FadeOut(reticle),
            Create(grid_lines),
            run_time=1.5,
        )
        self.wait(1.8)

        # Zoom back out to full frame for the central mystery question
        self.play(
            self.camera.frame.animate.set(width=14.222).move_to(ORIGIN),
            pixel_crop.animate.set_opacity(0.22).scale(1.6).move_to(ORIGIN),
            grid_lines.animate.set_stroke(opacity=0.08).scale(1.6).move_to(ORIGIN),
            run_time=2.2,
            rate_func=smooth,
        )

        # --- 0:28 - 0:35 | Pose the Core Question ---
        question_box = RoundedRectangle(
            width=10.8,
            height=2.2,
            corner_radius=0.18,
            fill_color=BG_PANEL,
            fill_opacity=0.90,
            stroke_color=COLOR_CYAN,
            stroke_width=2.0,
        )
        question_text = Text(
            "How can we represent this image\nusing far fewer numbers?",
            font_size=34,
            color=COLOR_WHITE,
            weight="BOLD",
            line_spacing=1.2,
        )
        question_sub = Text(
            f"{m*n:,} grayscale pixels  ->  Can linear algebra compress the essence?",
            font_size=20,
            color=COLOR_CYAN,
        )
        q_group = VGroup(question_text, question_sub).arrange(DOWN, buff=0.24)
        q_group.move_to(question_box)
        overlay = VGroup(question_box, q_group).move_to(ORIGIN)

        self.play(FadeIn(overlay, scale=0.95), run_time=1.2)
        self.wait(3.2)
        self.play(
            FadeOut(overlay),
            FadeOut(pixel_crop),
            FadeOut(grid_lines),
            FadeOut(particles),
            run_time=1.1,
        )
