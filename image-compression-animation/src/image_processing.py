"""Image loading, grayscale matrix conversion, and fallback procedural image generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


DEFAULT_TARGET_WIDTH = 320
DEFAULT_TARGET_HEIGHT = 180


@dataclass(frozen=True)
class ImageData:
    """Container for original color image, grayscale matrix A, and metadata."""

    source_path: Path
    color_array: np.ndarray  # shape (H, W, 3), uint8
    gray_array: np.ndarray  # shape (H, W), uint8
    matrix: np.ndarray  # shape (m, n), float64 in [0, 255]
    height: int  # m
    width: int  # n
    original_file_bytes: int
    illustrative_raw_bytes: int


def generate_fallback_photograph(
    output_path: Path,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Create a crisp procedural high-contrast color landscape if no asset exists.

    Used automatically when no photograph is found in assets/ so every scene
    remains runnable out of the box.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float64)
    nx = x_grid / width
    ny = y_grid / height

    # Deep twilight gradient sky
    sky_r = 12 + 45 * (1.0 - ny) ** 2 + 70 * np.exp(-((ny - 0.52) ** 2) / 0.015) * (1.0 - nx * 0.4)
    sky_g = 22 + 75 * (1.0 - ny) ** 1.5 + 55 * np.exp(-((ny - 0.50) ** 2) / 0.02)
    sky_b = 52 + 120 * (1.0 - ny) * (0.6 + 0.4 * np.sin(nx * np.pi))

    # Smooth volumetric clouds
    cloud = (
        0.5
        + 0.3 * np.sin(nx * 9.0 + np.cos(ny * 7.0))
        + 0.2 * np.cos(nx * 21.0 - ny * 11.0)
    )
    cloud_band = np.exp(-((ny - 0.36) ** 2) / 0.018) * np.clip(cloud, 0.0, 1.0)
    sky_r += 55 * cloud_band
    sky_g += 65 * cloud_band
    sky_b += 80 * cloud_band

    canvas = np.stack([sky_r, sky_g, sky_b], axis=-1)

    # Distant mountain range ridge
    ridge1 = 0.56 - 0.14 * np.sin(nx * 5.2 + 0.4) - 0.05 * np.abs(np.sin(nx * 19.0)) - 0.02 * np.sin(nx * 47.0)
    mask1 = ny >= ridge1
    shade1 = np.clip((ny - ridge1) * 4.5, 0.0, 1.0)
    m1_color = np.stack(
        [24 + 28 * (1 - shade1), 40 + 42 * (1 - shade1), 74 + 55 * (1 - shade1)],
        axis=-1,
    )
    canvas = np.where(mask1[..., None], m1_color, canvas)

    # Foreground jagged peaks
    ridge2 = 0.68 - 0.18 * np.exp(-((nx - 0.72) ** 2) / 0.025) - 0.04 * np.abs(np.cos(nx * 29.0))
    ridge2 -= 0.12 * np.exp(-((nx - 0.22) ** 2) / 0.03)
    mask2 = ny >= ridge2
    facet = 0.5 + 0.5 * np.sin(nx * 85.0 + ny * 40.0)
    m2_color = np.stack(
        [15 + 38 * facet, 26 + 52 * facet, 45 + 68 * facet],
        axis=-1,
    )
    canvas = np.where(mask2[..., None], m2_color, canvas)

    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)

    # Architectural observatory silhouette with warm glowing windows on left ridge
    base_x, base_y = int(width * 0.20), int(height * 0.54)
    poly = [
        (base_x - 110, base_y + 70),
        (base_x - 120, base_y),
        (base_x - 30, base_y - 55),
        (base_x + 95, base_y - 25),
        (base_x + 125, base_y + 70),
    ]
    draw.polygon(poly, fill=(18, 24, 38), outline=(120, 195, 235))
    for i in range(5):
        wx = base_x - 85 + i * 38
        wy = base_y + 8
        draw.rectangle([wx, wy, wx + 24, wy + 34], fill=(245, 198, 110), outline=(255, 235, 180))

    # Crisp celestial moon / light source for dynamic range
    draw.ellipse([int(width * 0.78), int(height * 0.14), int(width * 0.83), int(height * 0.23)], fill=(235, 246, 255))
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img.save(output_path, format="PNG")
    return output_path


def resolve_asset_image(assets_dir: Path) -> Path:
    """Find a photograph in assets/ or generate a fallback image if missing."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    preferred = [
        assets_dir / "sample_photo.png",
        assets_dir / "sample_photo.jpg",
        assets_dir / "sample_photo.jpeg",
    ]
    for path in preferred:
        if path.exists() and path.is_file():
            return path

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        matches = sorted(assets_dir.glob(ext))
        if matches:
            return matches[0]

    return generate_fallback_photograph(assets_dir / "sample_photo.png")


def load_image_matrix(
    image_path: Path | None = None,
    assets_dir: Path | None = None,
    target_size: Tuple[int, int] = (DEFAULT_TARGET_WIDTH, DEFAULT_TARGET_HEIGHT),
) -> ImageData:
    """Load an image with Pillow, resize for SVD performance, and build matrix A.

    Args:
        image_path: Optional explicit path to a photograph.
        assets_dir: Directory to search or create fallback image if image_path is None.
        target_size: (width, height) tuple used for numerical matrix A (m=height, n=width).

    Returns:
        ImageData with color RGB array, grayscale uint8 array, and float64 matrix A.
    """
    if image_path is None:
        if assets_dir is None:
            assets_dir = Path(__file__).resolve().parent.parent / "assets"
        image_path = resolve_asset_image(assets_dir)
    elif not image_path.exists():
        image_path = generate_fallback_photograph(image_path)

    original_file_bytes = image_path.stat().st_size

    with Image.open(image_path) as img:
        rgb_full = img.convert("RGB")
        # Resize smoothly using Lanczos resampling to keep matrix dimensions fast and clean
        rgb_resized = rgb_full.resize(target_size, Image.Resampling.LANCZOS)
        gray_resized = rgb_resized.convert("L")

    color_array = np.asarray(rgb_resized, dtype=np.uint8)
    gray_array = np.asarray(gray_resized, dtype=np.uint8)
    matrix = gray_array.astype(np.float64)

    height, width = matrix.shape
    illustrative_raw_bytes = height * width  # 1 byte per grayscale pixel

    return ImageData(
        source_path=image_path,
        color_array=color_array,
        gray_array=gray_array,
        matrix=matrix,
        height=height,
        width=width,
        original_file_bytes=original_file_bytes,
        illustrative_raw_bytes=illustrative_raw_bytes,
    )


def extract_matrix_patch(
    matrix: np.ndarray,
    row_start: int | None = None,
    col_start: int | None = None,
    patch_size: int = 8,
) -> Tuple[np.ndarray, int, int]:
    """Extract an integer-valued sub-matrix patch with high contrast for visualization."""
    m, n = matrix.shape
    if row_start is None:
        row_start = max(0, m // 2 - patch_size // 2)
    if col_start is None:
        col_start = max(0, n // 4 - patch_size // 2)

    row_start = min(max(0, row_start), m - patch_size)
    col_start = min(max(0, col_start), n - patch_size)
    patch = matrix[row_start : row_start + patch_size, col_start : col_start + patch_size]
    return np.rint(patch).astype(np.int64), row_start, col_start
