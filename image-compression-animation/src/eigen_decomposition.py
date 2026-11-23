"""Eigendecomposition of the symmetric Gram matrix A^T A."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EigenResult:
    """Verified eigendecomposition of the symmetric matrix A^T A = V Lambda V^T."""

    gram_matrix: np.ndarray  # A^T A, shape (n, n)
    eigenvalues: np.ndarray  # sorted descending, shape (n,)
    eigenvectors: np.ndarray  # columns are orthonormal eigenvectors V, shape (n, n)
    diagonal_lambda: np.ndarray  # Lambda = diag(eigenvalues), shape (n, n)
    symmetry_error: float  # ||(A^T A) - (A^T A)^T||_F
    orthonormality_error: float  # ||V^T V - I||_F
    reconstruction_error: float  # ||A^T A - V Lambda V^T||_F / ||A^T A||_F


def compute_ata_eigendecomposition(matrix_a: np.ndarray) -> EigenResult:
    """Compute the eigendecomposition of the real symmetric matrix A^T A.

    For any real m x n matrix A (even rectangular where A itself has no PDP^-1
    eigendecomposition), A^T A is an n x n real symmetric positive semi-definite matrix.
    By the Spectral Theorem, A^T A = V Lambda V^T with orthonormal columns in V.

    Args:
        matrix_a: Real image matrix A of shape (m, n).

    Returns:
        EigenResult with descending eigenvalues, orthonormal eigenvectors V,
        diagonal matrix Lambda, and numerical verification errors.
    """
    a = np.asarray(matrix_a, dtype=np.float64)
    gram = a.T @ a
    # Enforce exact floating-point symmetry prior to eigh
    gram_sym = 0.5 * (gram + gram.T)
    symmetry_error = float(np.linalg.norm(gram - gram.T, ord="fro"))

    # numpy.linalg.eigh returns eigenvalues in ascending order
    evals_asc, evecs_asc = np.linalg.eigh(gram_sym)

    # Sort in descending order so dominant directions come first
    idx = np.argsort(evals_asc)[::-1]
    eigenvalues = np.maximum(evals_asc[idx], 0.0)
    eigenvectors = evecs_asc[:, idx]

    diagonal_lambda = np.diag(eigenvalues)

    n = gram.shape[0]
    orthonormality_error = float(
        np.linalg.norm(eigenvectors.T @ eigenvectors - np.eye(n), ord="fro")
    )
    reconstructed_gram = eigenvectors @ diagonal_lambda @ eigenvectors.T
    gram_norm = float(np.linalg.norm(gram, ord="fro"))
    reconstruction_error = float(
        np.linalg.norm(gram - reconstructed_gram, ord="fro") / max(gram_norm, 1e-12)
    )

    return EigenResult(
        gram_matrix=gram,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        diagonal_lambda=diagonal_lambda,
        symmetry_error=symmetry_error,
        orthonormality_error=orthonormality_error,
        reconstruction_error=reconstruction_error,
    )
