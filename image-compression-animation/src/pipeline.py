"""End-to-end verified mathematical pipeline and asset exporter for all Manim scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import linalg as scipy_linalg

from .eigen_decomposition import EigenResult, compute_ata_eigendecomposition
from .image_processing import ImageData, extract_matrix_patch, load_image_matrix
from .reconstruction import ReconstructionResult, export_reconstructions
from .svd_compression import (
    SVDResult,
    compute_svd,
    verify_eigen_svd_relationship,
)


DEFAULT_RANKS: Tuple[int, ...] = (1, 5, 10, 20)


@dataclass(frozen=True)
class PipelineSummary:
    """Complete verified mathematical dataset consumed by Manim scenes."""

    image_data: ImageData
    eigen_result: EigenResult
    svd_result: SVDResult
    reconstructions: List[ReconstructionResult]
    eigen_svd_max_rel_diff: float
    scipy_svd_max_diff: float
    sample_patch: np.ndarray
    patch_origin: Tuple[int, int]
    export_dir: Path


def run_math_pipeline(
    project_root: Path | None = None,
    target_size: Tuple[int, int] = (320, 180),
    ranks: Tuple[int, ...] = DEFAULT_RANKS,
) -> PipelineSummary:
    """Execute and verify all NumPy/SciPy/Pillow calculations and export artifacts."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    assets_dir = project_root / "assets"
    export_dir = project_root / "output" / "computed_assets"
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load photograph & build grayscale matrix A (m x n)
    image_data = load_image_matrix(assets_dir=assets_dir, target_size=target_size)

    # Export crisp color and grayscale copies for Manim ImageMobject loading
    from PIL import Image

    upscale = (960, 540)
    Image.fromarray(image_data.color_array, mode="RGB").resize(
        upscale, Image.Resampling.LANCZOS
    ).save(export_dir / "original_color.png")

    # Save a high-contrast zoomed crop for Scene 1 & Scene 2 pixel reveal
    m, n = image_data.height, image_data.width
    patch, r0, c0 = extract_matrix_patch(
        image_data.matrix,
        row_start=int(m * 0.44),
        col_start=int(n * 0.36),
        patch_size=6,
    )

    # Also export a wider 24x14 pixel-grid crop around the glowing observatory for Scene 1/2
    crop_y0 = max(0, int(m * 0.36))
    crop_x0 = max(0, int(n * 0.24))
    crop_h, crop_w = 36, 64
    color_crop = image_data.color_array[crop_y0 : crop_y0 + crop_h, crop_x0 : crop_x0 + crop_w]
    gray_crop = image_data.gray_array[crop_y0 : crop_y0 + crop_h, crop_x0 : crop_x0 + crop_w]
    Image.fromarray(color_crop, mode="RGB").resize(
        (960, 540), Image.Resampling.NEAREST
    ).save(export_dir / "zoom_pixels_color.png")
    Image.fromarray(gray_crop, mode="L").resize(
        (960, 540), Image.Resampling.NEAREST
    ).save(export_dir / "zoom_pixels_gray.png")

    # 2. Compute A^T A eigendecomposition via numpy.linalg.eigh
    eigen_result = compute_ata_eigendecomposition(image_data.matrix)

    # 3. Compute SVD via numpy.linalg.svd and cross-check singular values with SciPy
    svd_result = compute_svd(image_data.matrix)
    scipy_s = scipy_linalg.svdvals(image_data.matrix)
    scipy_svd_max_diff = float(
        np.max(np.abs(svd_result.singular_values - scipy_s))
        / max(float(svd_result.singular_values[0]), 1e-12)
    )

    # 4. Verify lambda_i == sigma_i^2
    eigen_svd_max_rel_diff = verify_eigen_svd_relationship(
        eigen_result.eigenvalues, svd_result.singular_values
    )

    # 5. Compute and export rank-k and full-rank reconstructions
    rank_list = list(ranks) + [svd_result.full_rank]
    reconstructions = export_reconstructions(
        image_data.matrix,
        svd_result,
        ranks=rank_list,
        export_dir=export_dir,
        upscale_size=upscale,
    )

    # 6. Persist verified summary JSON for quick inspection and documentation
    summary_payload: Dict[str, object] = {
        "source_image": str(image_data.source_path.name),
        "matrix_shape": [image_data.height, image_data.width],
        "original_file_bytes": image_data.original_file_bytes,
        "original_matrix_values": image_data.height * image_data.width,
        "ata_symmetry_error": eigen_result.symmetry_error,
        "v_orthonormality_error": eigen_result.orthonormality_error,
        "ata_reconstruction_rel_error": eigen_result.reconstruction_error,
        "svd_reconstruction_rel_error": svd_result.svd_reconstruction_error,
        "eigen_svd_lambda_sigma2_max_rel_diff": eigen_svd_max_rel_diff,
        "scipy_svd_max_rel_diff": scipy_svd_max_diff,
        "top_10_singular_values": [float(x) for x in svd_result.singular_values[:10]],
        "top_10_eigenvalues": [float(x) for x in eigen_result.eigenvalues[:10]],
        "reconstructions": [
            {
                "rank_k": r.rank_k,
                "stored_values": r.stored_values,
                "original_values": r.original_values,
                "storage_ratio": round(r.storage_ratio, 6),
                "space_saved_percent": round(r.space_saved_percent, 2),
                "frobenius_relative_error": round(r.frobenius_relative_error, 6),
                "rmse": round(r.rmse, 3),
                "psnr_db": round(r.psnr_db, 2),
                "energy_retained_percent": round(r.energy_retained_percent, 2),
                "image_file": r.image_path.name if r.image_path else None,
            }
            for r in reconstructions
        ],
    }
    (export_dir / "metrics.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    return PipelineSummary(
        image_data=image_data,
        eigen_result=eigen_result,
        svd_result=svd_result,
        reconstructions=reconstructions,
        eigen_svd_max_rel_diff=eigen_svd_max_rel_diff,
        scipy_svd_max_diff=scipy_svd_max_diff,
        sample_patch=patch,
        patch_origin=(r0, c0),
        export_dir=export_dir,
    )


if __name__ == "__main__":
    result = run_math_pipeline()
    print(
        f"Verified pipeline on {result.image_data.source_path.name} "
        f"({result.image_data.height}x{result.image_data.width}):"
    )
    print(f"  max |lambda_i - sigma_i^2| / max = {result.eigen_svd_max_rel_diff:.3e}")
    print(f"  ||V^T V - I||_F                  = {result.eigen_result.orthonormality_error:.3e}")
    print(f"  ||A - U Sigma V^T||_F / ||A||_F  = {result.svd_result.svd_reconstruction_error:.3e}")
    for r in result.reconstructions:
        print(
            f"  k={r.rank_k:3d}: stored={r.stored_values:6d} / {r.original_values} "
            f"({r.space_saved_percent:6.2f}% saved) | "
            f"rel_err={r.frobenius_relative_error*100:5.2f}% | "
            f"PSNR={r.psnr_db:5.2f} dB"
        )
