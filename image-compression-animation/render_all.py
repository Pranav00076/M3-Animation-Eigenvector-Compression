"""Render all six Manim scenes in sequence and concatenate into a 1080p MP4."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_math_pipeline


SCENES = [
    ("scenes/scene01_intro.py", "Scene01Intro"),
    ("scenes/scene02_pixels.py", "Scene02Pixels"),
    ("scenes/scene03_eigenvectors.py", "Scene03Eigenvectors"),
    ("scenes/scene04_svd.py", "Scene04SVD"),
    ("scenes/scene05_reconstruction.py", "Scene05Reconstruction"),
    ("scenes/scene06_conclusion.py", "Scene06Conclusion"),
]

QUALITY_FLAGS = {
    "l": (["-ql"], "480p15"),
    "m": (["-qm"], "720p30"),
    "fhd": (["-r", "1920,1080", "--fps", "30"], "1080p30"),
    "h": (["-qh"], "1080p60"),
}


def find_rendered_mp4(media_dir: Path, scene_file: str, class_name: str, quality_dir: str) -> Path:
    stem = Path(scene_file).stem
    expected = media_dir / "videos" / stem / quality_dir / f"{class_name}.mp4"
    if expected.exists():
        return expected
    matches = list((media_dir / "videos").rglob(f"{class_name}.mp4"))
    if not matches:
        raise FileNotFoundError(f"Could not locate rendered MP4 for {class_name}")
    # Prefer most recently modified matching file
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render all scenes and assemble final MP4.")
    parser.add_argument(
        "--quality",
        choices=["l", "m", "fhd", "h"],
        default="fhd",
        help="Manim quality preset: l (480p15), m (720p30), fhd (1080p30, default), h (1080p60)",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        type=int,
        help="Optional 1-based scene numbers to render (e.g., --scenes 1 5).",
    )
    args = parser.parse_args()

    flags, quality_dir = QUALITY_FLAGS[args.quality]
    output_dir = PROJECT_ROOT / "output"
    media_dir = PROJECT_ROOT / "media"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1/3 Running verified NumPy/SciPy/Pillow mathematical pipeline...")
    summary = run_math_pipeline(project_root=PROJECT_ROOT)
    print(
        f"    Matrix A: {summary.image_data.height}x{summary.image_data.width} | "
        f"max |lambda_i - sigma_i^2|/max = {summary.eigen_svd_max_rel_diff:.2e}"
    )

    selected = (
        [SCENES[i - 1] for i in args.scenes]
        if args.scenes
        else SCENES
    )

    rendered_clips: list[Path] = []
    print(f"2/3 Rendering {len(selected)} Manim scene(s) at quality {' '.join(flags)} ({quality_dir})...")
    for scene_file, class_name in selected:
        cmd = [
            sys.executable,
            "-m",
            "manim",
            "render",
            *flags,
            "--media_dir",
            str(media_dir),
            scene_file,
            class_name,
        ]
        print(f"    -> {' '.join(cmd)}")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        clip = find_rendered_mp4(media_dir, scene_file, class_name, quality_dir)
        dest = output_dir / f"{Path(scene_file).stem}.mp4"
        shutil.copy2(clip, dest)
        rendered_clips.append(dest)
        print(f"       Saved {dest}")

    if len(rendered_clips) > 1:
        print("3/3 Concatenating scenes into final 16:9 documentary MP4...")
        concat_file = output_dir / "concat_list.txt"
        concat_file.write_text(
            "\n".join(f"file '{p.name}'" for p in rendered_clips) + "\n",
            encoding="utf-8",
        )
        final_mp4 = output_dir / "the_mathematics_behind_image_compression.mp4"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(final_mp4),
        ]
        subprocess.run(ffmpeg_cmd, cwd=output_dir, check=True)
        print(f"Done! Final video written to: {final_mp4}")
    else:
        print("3/3 Single-scene render complete.")


if __name__ == "__main__":
    main()
