"""IoT Gateway: Compressive Sensing Reconstruction Engine.

Reconstructs the original biomedical signal from M compressed measurements
using Orthogonal Matching Pursuit (OMP) in the Discrete Cosine Transform (DCT) basis.
"""

from typing import Tuple, Dict, Any
import numpy as np
from scipy import fft


def build_dct_basis(N: int) -> np.ndarray:
    """Generate orthonormal Discrete Cosine Transform (DCT-II) dictionary Psi."""
    Psi = np.zeros((N, N))
    for k in range(N):
        alpha = np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)
        for n in range(N):
            Psi[n, k] = alpha * np.cos(np.pi * (2 * n + 1) * k / (2.0 * N))
    return Psi


def omp_reconstruct(
    y: np.ndarray,
    Phi: np.ndarray,
    Psi: np.ndarray,
    max_iter: int = 40,
    tol: float = 1e-4
) -> np.ndarray:
    """Orthogonal Matching Pursuit (OMP) algorithm for sparse recovery.

    Solves: min ||s||_0  s.t. ||y - Phi * Psi * s||_2 <= tol
    """
    M, N = Phi.shape
    # Equivalent dictionary matrix A = Phi * Psi
    A = np.dot(Phi, Psi)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A_normalized = A / col_norms

    residual = y.copy()
    support = []
    s = np.zeros(N)

    for _ in range(min(max_iter, M)):
        # 1. Match: find column of normalized A most correlated with residual
        correlations = np.abs(np.dot(A_normalized.T, residual))
        for idx in support:
            correlations[idx] = -1.0
        best_idx = int(np.argmax(correlations))
        support.append(best_idx)

        # 2. Project: solve least-squares on chosen support
        A_sub = A[:, support]
        s_sub, _, _, _ = np.linalg.lstsq(A_sub, y, rcond=None)

        # 3. Update residual
        residual = y - np.dot(A_sub, s_sub)

        if np.linalg.norm(residual) < tol:
            break

    s[support] = s_sub
    # Reconstruct original time-domain signal: x = Psi * s
    x_rec = np.dot(Psi, s)
    return x_rec


def compute_reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """Computes clinical fidelity metrics: PRD (%) and SNR (dB).

    PRD (Percentage Root-Mean-Square Difference) < 2% indicates 'very good / clinically identical' recovery.
    """
    diff = original - reconstructed
    prd = float(np.sqrt(np.sum(diff ** 2) / (np.sum(original ** 2) + 1e-8)) * 100.0)
    snr_db = float(10.0 * np.log10(np.sum(original ** 2) / (np.sum(diff ** 2) + 1e-8)))

    return {
        "prd_percent": prd,
        "snr_db": snr_db
    }
