"""Validation of the Python port.

1. finufft convention == Fessler's DTFT adjoint with n_shift = N/2 (brute-force sum,
   i.e. the operator that MATLAB's NUFFT(...)' approximates).
2. Literal port of the MATLAB loop == optimised incremental/parallel reconstruction.
Run:  python -m pytest -q tests   (or python tests/test_equivalence.py)
"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtmri_recon import ReconParams, reconstruct, reconstruct_reference, radial_trajectory, radial_dcf


def make_raw(T, C, N, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, C, N)) + 1j * rng.standard_normal((T, C, N))).astype(np.complex64)


def test_convention_vs_bruteforce_dtft():
    import finufft
    N, S, s = 24, 12, 3
    om1, om2 = radial_trajectory(N, S, s)
    X = (np.random.default_rng(1).standard_normal((2, S * N)) * (1 + 1j))
    n = np.arange(N) - N // 2                           # n - n_shift, n_shift = N/2
    ph = np.exp(1j * (om1.ravel()[:, None, None] * n[None, :, None]
                      + om2.ravel()[:, None, None] * n[None, None, :]))   # (M, N, N)
    ref = np.einsum("cm,mij->cij", X, ph)
    got = finufft.nufft2d1(om1.ravel(), om2.ravel(), X, (N, N), eps=1e-12, isign=+1)
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-10


def _run(N, S, s, C, T, channels=None, single=False, n_jobs=1, **kw):
    raw = make_raw(T, C, N)
    p = ReconParams(n=N, n_spokes=S, shots=s, channels=channels, single=single, eps=1e-9 if not single else 1e-5, **kw)
    ref = reconstruct_reference(raw, p)
    fast = np.array(reconstruct(raw, p, n_jobs=n_jobs, verbose=False))
    err = np.abs(fast - ref).max() / ref.max()
    return ref, fast, err


def test_incremental_equals_reference_double():
    _, _, err = _run(32, 20, 4, 5, 20 + 4 * 30 + 2)           # includes 'extra' trailing spokes
    assert err < 1e-6, err


def test_incremental_equals_reference_single_parallel_channels():
    _, _, err = _run(32, 20, 4, 6, 20 + 4 * 25 + 3, channels=[0, 2, 3, 5], single=True, n_jobs=2,
                     refresh_every=7)
    assert err < 1e-3, err


def test_central_crop_exact():
    _, _, err = _run(32, 20, 4, 4, 20 + 4 * 12 + 1, n_out=18)   # even crop around centre
    assert err < 1e-6, err


def test_single_frame_edge_case():
    _, _, err = _run(16, 8, 4, 2, 8 + 3)                       # numFrames == 1
    assert err < 1e-6, err


if __name__ == "__main__":
    test_convention_vs_bruteforce_dtft(); print("convention OK")
    test_incremental_equals_reference_double(); print("incremental == reference (double) OK")
    test_incremental_equals_reference_single_parallel_channels(); print("single/parallel/chan-subset OK")
    test_central_crop_exact(); print("central crop OK")
    test_single_frame_edge_case(); print("edge case OK")
