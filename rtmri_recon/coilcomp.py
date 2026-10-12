"""Optional SVD coil compression (approximate speed-up: fewer coils to transform)."""
from __future__ import annotations
import numpy as np


def coil_compression_matrix(raw, n_virtual: int, channels=None, n_spokes_used: int = 400,
                            n_center: int = 9) -> np.ndarray:
    """(K, C) matrix; apply as virtual = M @ physical.  Built from k-space centre samples."""
    T, Cfull, N = raw.shape
    sel = np.linspace(0, T - 1, min(n_spokes_used, T)).astype(int)
    c0 = N // 2
    blk = np.asarray(raw[sel])[:, :, c0 - n_center // 2: c0 + n_center // 2 + 1]
    if channels is not None:
        blk = blk[:, np.asarray(channels), :]
    A = blk.transpose(1, 0, 2).reshape(blk.shape[1], -1)       # (C, samples)
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    return np.ascontiguousarray(U[:, :n_virtual].conj().T)
