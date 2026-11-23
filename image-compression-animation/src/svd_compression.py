"""Singular Value Decomposition (SVD) and energy/storage analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SVDResult:
    """Full SVD factorization A = U Sigma V^T with singular value spectrum."""

    u: np.ndarray  # shape (m, r)
    singular_values: np.ndarray  # shape (r,), sorted descending
    vt: np.ndarray  # shape (r, n)
    sigma_matrix: np.ndarray  # diagonal matrix diag(singular_values), shape (r, r)
    full_rank: int  # r = min(m, n)
    original_shape: tuple[int, int]  # (m, n)
    cumulative_energy_ratio: np.ndarray  # sum_{i<=k} sigma_i^2 / sum_all sigma_i^2
    svd_reconstruction_error: float  # ||A - U Sigma V^T||_F / ||A||_F


def compute_svd(matrix_a: np.ndarray) -> SVDResult:
    """Compute economic SVD A = U Sigma V^T via numpy.linalg.svd.

    Args:
        matrix_a: Real m x n matrix A.

    Returns:
        SVDResult containing U, Sigma (descending singular values), V^T,
        cumulative Frobenius energy ratios, and reconstruction error check.
    """
    a = np.asarray(matrix_a, dtype=np.float64)
    m, n = a.shape
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    sigma_matrix = np.diag(s)

    squared = s**2
    total_energy = float(np.sum(squared))
    cumulative_energy_ratio = np.cumsum(squared) / max(total_energy, 1e-12)

    reconstructed = (u * s) @ vt
    a_norm = float(np.linalg.norm(a, ord="fro"))
    svd_reconstruction_error = float(
        np.linalg.norm(a - reconstructed, ord="fro") / max(a_norm, 1e-12)
    )

    return SVDResult(
        u=u,
        singular_values=s,
        vt=vt,
        sigma_matrix=sigma_matrix,
        full_rank=len(s),
        original_shape=(m, n),
        cumulative_energy_ratio=cumulative_energy_ratio,
        svd_reconstruction_error=svd_reconstruction_error,
    )


def verify_eigen_svd_relationship(
    eigenvalues: np.ndarray,
    singular_values: np.ndarray,
) -> float:
    """Return max relative difference between eigenvalues of A^T A and sigma_i^2.

    Mathematically, if A = U Sigma V^T, then:
        A^T A = V (Sigma^T Sigma) V^T = V Lambda V^T
    so lambda_i = sigma_i^2 for i = 1, ..., r.
    """
    r = min(len(eigenvalues), len(singular_values))
    lam = eigenvalues[:r]
    sig_sq = singular_values[:r] ** 2
    scale = max(float(np.max(sig_sq)), 1e-12)
    return float(np.max(np.abs(lam - sig_sq)) / scale)


def truncated_storage_count(m: int, n: int, k: int) -> int:
    """Return number of stored scalars for rank-k SVD: k * (m + n + 1)."""
    return int(k * (m + n + 1))


def original_storage_count(m: int, n: int) -> int:
    """Return number of stored scalars for full matrix A: m * n."""
    return int(m * n)
