"""Low-rank SVD image reconstruction, error metrics, and PNG export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from .svd_compression import (
    SVDResult,
    original_storage_count,
    truncated_storage_count,
)


@dataclass(frozen=True)
class ReconstructionResult:
    """Result for a single rank-k approximation A_k = U_k Sigma_k V_k^T."""

    rank_k: int
    matrix_approx: np.ndarray  # float64 raw A_k
    image_uint8: np.ndarray  # clipped [0, 255] uint8 image
    stored_values: int  # k(m + n + 1) or m*n when full rank
    original_values: int  # m * n
    storage_ratio: float  # stored_values / original_values
    space_saved_percent: float  # (1 - storage_ratio) * 100
    frobenius_relative_error: float  # ||A - A_k||_F / ||A||_F
    rmse: float  # root mean squared pixel error
    psnr_db: float  # peak signal-to-noise ratio in dB
    energy_retained_percent: float  # sum_{i<=k} sigma_i^2 / total * 100
    image_path: Path | None = None


def reconstruct_rank_k(
    original_matrix: np.ndarray,
    svd_result: SVDResult,
    k: int,
    is_full_rank: bool = False,
) -> ReconstructionResult:
    """Compute A_k = U_k Sigma_k V_k^T and exact quality & storage metrics."""
    m, n = original_matrix.shape
    r = svd_result.full_rank
    k_clamped = max(1, min(int(k), r))

    u_k = svd_result.u[:, :k_clamped]
    s_k = svd_result.singular_values[:k_clamped]
    vt_k = svd_result.vt[:k_clamped, :]

    approx = (u_k * s_k) @ vt_k
    clipped_uint8 = np.clip(np.rint(approx), 0, 255).astype(np.uint8)

    diff = original_matrix - approx
    orig_norm = float(np.linalg.norm(original_matrix, ord="fro"))
    fro_rel_error = float(np.linalg.norm(diff, ord="fro") / max(orig_norm, 1e-12))

    mse = float(np.mean((original_matrix - clipped_uint8.astype(np.float64)) ** 2))
    rmse = float(np.sqrt(mse))
    psnr_db = float(20.0 * np.log10(255.0 / max(rmse, 1e-12))) if rmse > 0 else 99.9

    orig_vals = original_storage_count(m, n)
    if is_full_rank or k_clamped == r:
        stored_vals = orig_vals
    else:
        stored_vals = truncated_storage_count(m, n, k_clamped)

    storage_ratio = stored_vals / orig_vals
    space_saved_percent = (1.0 - storage_ratio) * 100.0
    energy_retained_percent = float(svd_result.cumulative_energy_ratio[k_clamped - 1] * 100.0)

    return ReconstructionResult(
        rank_k=k_clamped,
        matrix_approx=approx,
        image_uint8=clipped_uint8,
        stored_values=stored_vals,
        original_values=orig_vals,
        storage_ratio=storage_ratio,
        space_saved_percent=space_saved_percent,
        frobenius_relative_error=fro_rel_error,
        rmse=rmse,
        psnr_db=psnr_db,
        energy_retained_percent=energy_retained_percent,
    )


def export_reconstructions(
    original_matrix: np.ndarray,
    svd_result: SVDResult,
    ranks: Iterable[int],
    export_dir: Path,
    upscale_size: tuple[int, int] = (960, 540),
) -> List[ReconstructionResult]:
    """Reconstruct and export PNG images for each requested rank k plus full rank."""
    export_dir.mkdir(parents=True, exist_ok=True)
    results: List[ReconstructionResult] = []

    # Export crisp original grayscale reference
    orig_uint8 = np.clip(np.rint(original_matrix), 0, 255).astype(np.uint8)
    orig_img = Image.fromarray(orig_uint8, mode="L").resize(
        upscale_size, Image.Resampling.NEAREST
    )
    orig_img.save(export_dir / "original_gray.png")

    for rank in ranks:
        is_full = rank >= svd_result.full_rank
        res = reconstruct_rank_k(original_matrix, svd_result, rank, is_full_rank=is_full)
        tag = "full" if is_full else f"k_{res.rank_k:03d}"
        out_path = export_dir / f"recon_{tag}.png"
        img = Image.fromarray(res.image_uint8, mode="L").resize(
            upscale_size, Image.Resampling.NEAREST
        )
        img.save(out_path)
        results.append(
            ReconstructionResult(
                rank_k=res.rank_k,
                matrix_approx=res.matrix_approx,
                image_uint8=res.image_uint8,
                stored_values=res.stored_values,
                original_values=res.original_values,
                storage_ratio=res.storage_ratio,
                space_saved_percent=res.space_saved_percent,
                frobenius_relative_error=res.frobenius_relative_error,
                rmse=res.rmse,
                psnr_db=res.psnr_db,
                energy_retained_percent=res.energy_retained_percent,
                image_path=out_path,
            )
        )

    return results
