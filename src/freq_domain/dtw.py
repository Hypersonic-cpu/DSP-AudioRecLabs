"""
Dynamic Time Warping (DTW) algorithm for template matching.
Core algorithm for Experiment 3.
"""
import torch
import numpy as np
from numba import jit


def dtw_distance(seq1: torch.Tensor, seq2: torch.Tensor) -> float:
    """
    Compute DTW distance between two sequences.

    Uses dynamic programming to find optimal alignment.
    Formula: D(i,j) = d(i,j) + min{D(i-1,j), D(i,j-1), D(i-1,j-1)}

    Args:
        seq1: Test sequence [N, feature_dim]
        seq2: Reference template [M, feature_dim]

    Returns:
        DTW distance (float)
    """
    # Convert to numpy for faster computation
    if torch.is_tensor(seq1):
        seq1 = seq1.cpu().numpy()
    if torch.is_tensor(seq2):
        seq2 = seq2.cpu().numpy()

    return dtw_distance_fast(seq1, seq2)


@jit(nopython=True)
def _dtw_core(dist_matrix: np.ndarray, n: int, m: int) -> float:
    """
    JIT-compiled DTW core computation.
    """
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[n, m]


def dtw_distance_fast(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Fast DTW distance computation using numpy + numba JIT.

    Args:
        seq1: Test sequence [N, feature_dim]
        seq2: Reference template [M, feature_dim]

    Returns:
        DTW distance (float)
    """
    n, m = seq1.shape[0], seq2.shape[0]

    # Compute distance matrix (vectorized)
    dist_matrix = np.sqrt(((seq1[:, np.newaxis, :] - seq2[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Call JIT-compiled core
    return _dtw_core(dist_matrix, n, m)
