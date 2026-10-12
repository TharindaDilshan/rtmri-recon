"""Sliding-window radial reconstruction (sum-of-squares over coils).

Equivalent of the frame loop in RTMRIProcess.m, but:

* the adjoint NUFFT is linear, so when the window slides by `shots` spokes we
  only grid the *change* in k-space (2*new - old for the replaced spokes) and
  add it to the running per-coil images instead of regridding all
  `n_spokes` spokes of all coils for every frame;
* frames are split into contiguous chunks that run in parallel processes
  (each chunk initialises itself from the exact window state at its first
  frame, so results do not depend on the chunking);
* the NUFFT is finufft (CPU, multithreaded) instead of Fessler's sparse
  matrix; both approximate the same DTFT (x[n] = sum_m X_m exp(+i om_m (n - N/2))).
"""
from __future__ import annotations

import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

from .trajectory import radial_trajectory, radial_dcf


# --------------------------------------------------------------------------- #
# Parameters / window bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class ReconParams:
    n: int                                   # base resolution (NCol) == image size
    n_spokes: int                            # spokes per full window (NLin)
    shots: int                               # spokes per shot == window step (NLin / NSeg)
    channels: Optional[Sequence[int]] = None  # 0-based coil subset (None = all)
    new_gain: float = 2.0                    # MATLAB multiplies replaced spokes by 2
    eps: float = 1e-5                        # NUFFT tolerance (finufft)
    single: bool = True                      # complex64 compute (2x faster); False = complex128
    cc_matrix: Optional[np.ndarray] = None   # optional (K, n_selected_channels) coil compression
    refresh_every: int = 256                 # exact regrid every N frames to bound drift
    upsampfac: Optional[float] = None        # finufft upsampling (None = auto; 1.25 is faster)
    n_out: Optional[int] = None              # output only the central n_out x n_out pixels (exact, much
                                             # faster); 146 == the crop MATLAB uses for the video

    @property
    def size_out(self) -> int:
        return self.n if self.n_out is None else int(self.n_out)


def window_plan(n_total: int, n_spokes: int, shots: int):
    """Number of frames and the 'extra' trailing spokes (MATLAB numFrames / extraSpokes)."""
    n_frames = int(math.ceil((n_total - n_spokes) / shots))
    if n_frames < 1:
        raise ValueError(f"need more than n_spokes={n_spokes} spokes, have {n_total}")
    extra = n_spokes + (n_frames - 1) * shots + np.arange(
        n_total - ((n_frames - 1) * shots + n_spokes))
    return n_frames, extra


def update_spokes(u: int, n_spokes: int, shots: int, n_frames: int, extra: np.ndarray):
    """Global (0-based) spoke indices written by update u (frame u -> frame u+1)."""
    idx = n_spokes + u * shots + np.arange(shots)
    if u == n_frames - 2:                   # MATLAB: I+1 == numFrames branch
        idx = np.concatenate([idx, extra])
    return idx


def _dedup_last(slots: np.ndarray, idx: np.ndarray):
    """MATLAB assignment k_space(:,:,slots) = ... keeps the last write per slot."""
    if len(np.unique(slots)) == len(slots):
        return slots, idx
    _, first_rev = np.unique(slots[::-1], return_index=True)
    keep = np.sort(len(slots) - 1 - first_rev)
    return slots[keep], idx[keep]


def window_state(f: int, p: ReconParams, n_frames: int, extra: np.ndarray):
    """Which raw spoke (src) and gain currently occupies each window slot at frame f."""
    src = np.arange(p.n_spokes)
    gain = np.ones(p.n_spokes)
    for u in range(f):
        idx = update_spokes(u, p.n_spokes, p.shots, n_frames, extra)
        slots = idx % p.n_spokes
        slots, idx = _dedup_last(slots, idx)
        src[slots] = idx
        gain[slots] = p.new_gain
    return src, gain


# --------------------------------------------------------------------------- #
# NUFFT engine
# --------------------------------------------------------------------------- #
class _Gridder:
    """Adjoint NUFFT (non-uniform k-space -> n x n image) for n_trans coils at once."""

    def __init__(self, p: ReconParams, n_trans: int, nthreads: int):
        import finufft
        self.n = p.n
        self.single = p.single
        self.ft = np.float32 if p.single else np.float64
        self.ct = np.complex64 if p.single else np.complex128
        kw = dict(nthreads=nthreads)
        if p.upsampfac is not None:
            kw["upsampfac"] = p.upsampfac
        m = p.size_out
        self.plan = finufft.Plan(1, (m, m), n_trans=n_trans, eps=p.eps, isign=+1,
                                 dtype="complex64" if p.single else "complex128", **kw)

    def __call__(self, om1: np.ndarray, om2: np.ndarray, data: np.ndarray) -> np.ndarray:
        self.plan.setpts(np.ascontiguousarray(om1, self.ft), np.ascontiguousarray(om2, self.ft))
        return self.plan.execute(np.ascontiguousarray(data, self.ct))


def _load_spokes(raw, idx: np.ndarray, p: ReconParams, scale: np.ndarray, dcf: np.ndarray,
                 ch: Optional[np.ndarray]) -> np.ndarray:
    """raw[(T, C, N)] -> (C', len(idx)*N) data, multiplied by scale[spoke] * dcf[sample]."""
    blk = np.asarray(raw[idx])                       # (n, C, N) copy
    if ch is not None:
        blk = blk[:, ch, :]
    ct = np.complex64 if p.single else np.complex128
    blk = blk.astype(ct, copy=False)
    blk = blk * (scale[:, None, None] * dcf[None, None, :]).astype(np.float32 if p.single else np.float64)
    blk = np.ascontiguousarray(blk.transpose(1, 0, 2)).reshape(blk.shape[1], -1)
    if p.cc_matrix is not None:
        blk = p.cc_matrix.astype(ct, copy=False) @ blk
    return blk


def _sumsq(acc: np.ndarray, out: np.ndarray):
    """out = sqrt(sum_c |acc_c|^2) without temporaries."""
    v = acc.view(np.float32 if acc.dtype == np.complex64 else np.float64)
    v = v.reshape(acc.shape + (2,))
    np.sqrt(np.einsum("cijk,cijk->ij", v, v), out=out, casting="unsafe")


def _n_out_coils(p: ReconParams, n_ch_raw: int) -> int:
    if p.cc_matrix is not None:
        return p.cc_matrix.shape[0]
    return n_ch_raw if p.channels is None else len(p.channels)


def _recon_range(raw, p: ReconParams, f0: int, f1: int, out, nthreads: int,
                 n_frames: int, extra: np.ndarray):
    N, S = p.n, p.n_spokes
    om1, om2 = radial_trajectory(N, S, p.shots)
    dcf = radial_dcf(N, S) / N                       # NUFFT operator divides by sqrt(N*N) = N
    ch = None if p.channels is None else np.asarray(p.channels)
    C = _n_out_coils(p, raw.shape[1])
    grid = _Gridder(p, C, nthreads)
    ct = grid.ct

    src, gain = window_state(f0, p, n_frames, extra)

    def full_window():
        slots = np.arange(S)
        data = _load_spokes(raw, src, p, gain, dcf, ch)
        return grid(om1[slots].ravel(), om2[slots].ravel(), data).astype(ct, copy=False)

    acc = full_window()
    for f in range(f0, f1):
        if f > f0 and p.refresh_every and (f - f0) % p.refresh_every == 0:
            acc = full_window()
        _sumsq(acc, out[f])
        if f + 1 < n_frames and f + 1 < f1:           # advance window f -> f+1
            idx = update_spokes(f, S, p.shots, n_frames, extra)
            slots, idx = _dedup_last(idx % S, idx)
            new = _load_spokes(raw, idx, p, np.full(len(idx), p.new_gain), dcf, ch)
            old = _load_spokes(raw, src[slots], p, gain[slots], dcf, ch)
            acc += grid(om1[slots].ravel(), om2[slots].ravel(), new - old)
            src[slots] = idx
            gain[slots] = p.new_gain


def _worker(raw_path, out_path, pdict, f0, f1, nthreads, n_frames, extra):
    raw = np.load(raw_path, mmap_mode="r")
    out = np.load(out_path, mmap_mode="r+")
    _recon_range(raw, ReconParams(**pdict), f0, f1, out, nthreads, n_frames, extra)
    out.flush()
    return f0, f1


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def reconstruct(raw, p: ReconParams, out_path: Optional[str] = None, n_jobs: Optional[int] = None,
                chunk_frames: Optional[int] = None, verbose: bool = True) -> np.ndarray:
    """Reconstruct all frames.

    raw       : path to a .npy file, or array, shaped (n_acquired_spokes, n_coils, n)
    returns   : float32 array (n_frames, n, n); if out_path is given it is a memmap on disk
    """
    tmp_raw = None
    if isinstance(raw, (str, os.PathLike)):
        raw_path = str(raw)
        raw_arr = np.load(raw_path, mmap_mode="r")
    else:
        raw_arr = raw
        raw_path = None
    n_frames, extra = window_plan(raw_arr.shape[0], p.n_spokes, p.shots)
    n_jobs = n_jobs or os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, n_frames))

    if out_path is None and n_jobs > 1:
        out_path = os.path.join(tempfile.mkdtemp(prefix="rtmri_"), "frames.npy")
    if out_path is not None:
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                        shape=(n_frames, p.size_out, p.size_out))
    else:
        out = np.zeros((n_frames, p.size_out, p.size_out), np.float32)

    if n_jobs == 1:
        _recon_range(raw_arr, p, 0, n_frames, out, os.cpu_count() or 1, n_frames, extra)
        return out

    if raw_path is None:                              # workers need a file to memmap
        tmp_raw = os.path.join(tempfile.mkdtemp(prefix="rtmri_"), "raw.npy")
        np.save(tmp_raw, raw_arr)
        raw_path = tmp_raw
    out.flush()
    del out
    if chunk_frames is None:
        chunk_frames = max(64, math.ceil(n_frames / (n_jobs * 3)))
    ranges = [(a, min(a + chunk_frames, n_frames)) for a in range(0, n_frames, chunk_frames)]
    import multiprocessing as mp
    pdict = asdict(p)
    done = 0
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_worker, raw_path, out_path, pdict, a, b, 1, n_frames, extra)
                for a, b in ranges]
        for fu in as_completed(futs):
            a, b = fu.result()
            done += b - a
            if verbose:
                print(f"  frames {a}-{b} done ({done}/{n_frames})", flush=True)
    if tmp_raw:
        os.remove(tmp_raw)
    return np.load(out_path, mmap_mode="r")


def reconstruct_reference(raw, p: ReconParams) -> np.ndarray:
    """Literal port of the MATLAB loop (full regrid every frame, double precision).

    Slow; used only to validate `reconstruct`.
    """
    raw = np.asarray(raw)
    N, S, s = p.n, p.n_spokes, p.shots
    n_frames, extra = window_plan(raw.shape[0], S, s)
    om1, om2 = radial_trajectory(N, S, s)
    dcf = radial_dcf(N, S)
    ch = np.arange(raw.shape[1]) if p.channels is None else np.asarray(p.channels)
    import finufft
    kspace = raw[:S].astype(np.complex128)           # (S, Cfull, N)
    frames = np.zeros((n_frames, N, N))
    for I in range(1, n_frames + 1):
        data = kspace[:, ch, :] * dcf[None, None, :]                 # (S, C, N)
        data = np.ascontiguousarray(data.transpose(1, 0, 2)).reshape(len(ch), -1)
        img = finufft.nufft2d1(om1.ravel(), om2.ravel(), data, (N, N), eps=1e-12, isign=+1) / N
        frames[I - 1] = np.sqrt((np.abs(img) ** 2).sum(0))
        if I + 1 <= n_frames:
            idx = S + (I - 1) * s + np.arange(s)
            if I + 1 == n_frames:
                idx = np.concatenate([idx, extra])
            kspace[idx % S] = p.new_gain * raw[idx]
    if p.n_out is not None:                          # central crop, same pixel convention as the NUFFT modes
        a = N // 2 - p.n_out // 2
        frames = frames[:, a:a + p.n_out, a:a + p.n_out]
    return frames
