"""Pytest verification suite for eigendecomposition, SVD, and reconstruction math."""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eigen_decomposition import compute_ata_eigendecomposition
from src.image_processing import extract_matrix_patch, generate_fallback_photograph, load_image_matrix
from src.pipeline import run_math_pipeline
from src.reconstruction import reconstruct_rank_k
from src.svd_compression import (
    compute_svd,
    original_storage_count,
    truncated_storage_count,
    verify_eigen_svd_relationship,
)


def test_fallback_image_and_matrix_loading(tmp_path: Path) -> None:
    fallback_path = tmp_path / "missing_photo.png"
    data = load_image_matrix(image_path=fallback_path, target_size=(80, 45))
    assert data.source_path.exists()
    assert data.matrix.shape == (45, 80)
    assert data.matrix.dtype == np.float64
    assert 0.0 <= data.matrix.min() <= data.matrix.max() <= 255.0


def test_symmetric_ata_and_eigen_svd_identity(tmp_path: Path) -> None:
    img_path = generate_fallback_photograph(tmp_path / "sample.png", width=160, height=90)
    data = load_image_matrix(image_path=img_path, target_size=(64, 36))

    eigen = compute_ata_eigendecomposition(data.matrix)
    svd = compute_svd(data.matrix)

    assert eigen.symmetry_error < 1e-9
    assert eigen.orthonormality_error < 1e-10
    assert eigen.reconstruction_error < 1e-10
    assert svd.svd_reconstruction_error < 1e-10

    max_diff = verify_eigen_svd_relationship(eigen.eigenvalues, svd.singular_values)
    assert max_diff < 1e-10


def test_low_rank_error_decreases_and_storage_formula(tmp_path: Path) -> None:
    img_path = generate_fallback_photograph(tmp_path / "sample.png", width=160, height=90)
    data = load_image_matrix(image_path=img_path, target_size=(120, 80))
    svd = compute_svd(data.matrix)

    ranks = [1, 5, 10, 20, svd.full_rank]
    errors = []
    for k in ranks:
        res = reconstruct_rank_k(data.matrix, svd, k, is_full_rank=(k == svd.full_rank))
        errors.append(res.frobenius_relative_error)
        if k < svd.full_rank:
            assert res.stored_values == truncated_storage_count(80, 120, k)
        else:
            assert res.stored_values == original_storage_count(80, 120)

    # Frobenius error must be monotonically non-increasing with rank k
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1] - 1e-12
    assert errors[-1] < 1e-10


def test_extract_patch_dimensions() -> None:
    mat = np.arange(20 * 30, dtype=np.float64).reshape(20, 30)
    patch, r0, c0 = extract_matrix_patch(mat, row_start=5, col_start=10, patch_size=6)
    assert patch.shape == (6, 6)
    assert (r0, c0) == (5, 10)
    assert patch.dtype == np.int64


def test_full_pipeline_exports(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    generate_fallback_photograph(assets_dir / "sample_photo.png", width=160, height=90)
    summary = run_math_pipeline(project_root=tmp_path, target_size=(120, 68))
    assert (summary.export_dir / "metrics.json").exists()
    assert (summary.export_dir / "original_color.png").exists()
    assert (summary.export_dir / "original_gray.png").exists()
    assert summary.eigen_svd_max_rel_diff < 1e-10
    assert summary.scipy_svd_max_diff < 1e-10
