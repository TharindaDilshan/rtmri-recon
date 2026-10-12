"""Siemens TWIX (.dat) reader -> on-disk (T, coils, NCol) complex64 .npy cache.

Uses pymapVBVD (Python port of the mapVBVD.m used by the MATLAB code).
The raw data are written spoke-major so each acquired spoke is contiguous,
which is what the reconstruction reads, and are processed in chunks so the
whole file never has to sit in RAM at once.
"""
from __future__ import annotations

import os
import numpy as np


def read_twix_to_npy(dat_path: str, npy_path: str, chunk: int = 2048, verbose: bool = True):
    """Returns dict(n_col, n_cha, n_spokes (NLin), n_seg (NSeg), shots, n_total)."""
    from mapvbvd import mapVBVD
    twix = mapVBVD(dat_path, quiet=True)
    if isinstance(twix, list):                      # multi-RAID files: MATLAB used twix{2}
        twix = twix[-1]
    img = twix.image
    img.flagRemoveOS = False
    img.flagDoAverage = True
    info = dict(n_col=int(img.NCol), n_cha=int(img.NCha), n_spokes=int(img.NLin),
                n_seg=int(img.NSeg), n_rep=int(img.NRep))
    info["shots"] = info["n_spokes"] // info["n_seg"]
    mem = np.asarray(img.memPos)
    n_total = len(mem)
    info["n_total"] = n_total
    out = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.complex64,
                                    shape=(n_total, info["n_cha"], info["n_col"]))
    for a in range(0, n_total, chunk):
        b = min(a + chunk, n_total)
        blk = img.readData(mem[a:b])                # (NCol, NCha, b-a), same as unsorted()
        out[a:b] = np.asarray(blk).transpose(2, 1, 0)
        if verbose:
            print(f"    read spokes {b}/{n_total}", end="\r", flush=True)
    out.flush()
    return info
