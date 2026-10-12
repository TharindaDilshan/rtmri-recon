"""Batch driver: every *.dat in a folder -> frames (.npy / optional .mat) + .mp4."""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from .recon import ReconParams, reconstruct


def process_file(dat_path: str, out_dir: str, a):
    """.dat -> <out_dir>/<name>.mp4. All intermediates live in a temp folder that is removed afterwards."""
    import gc
    import tempfile
    t0 = time.time()
    name = os.path.splitext(os.path.basename(dat_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    is_npy = dat_path.lower().endswith(".npy")
    # ignore_cleanup_errors: Windows can briefly keep memory-mapped files locked
    with tempfile.TemporaryDirectory(prefix="rtmri_", ignore_cleanup_errors=True) as tmp:
        if is_npy:
            if not (a.n_spokes and a.shots):
                raise SystemExit("for a cached .npy input, give --n-spokes (NLin) and --shots (NLin/NSeg)")
            raw_npy = dat_path
            info = dict(n_col=np.load(raw_npy, mmap_mode="r").shape[2], n_spokes=a.n_spokes, shots=a.shots)
        else:
            from .reader import read_twix_to_npy
            print(f"[{name}] reading TWIX ...")
            raw_npy = os.path.join(tmp, "raw.npy")
            info = read_twix_to_npy(dat_path, raw_npy)
        t_read = time.time() - t0
        n_spokes = a.n_spokes or info["n_spokes"]
        shots = a.shots or info["shots"]
        channels = None if not a.channels else [c - 1 for c in a.channels]   # CLI is 1-based like MATLAB

        cc = None
        if a.coil_compress:
            from .coilcomp import coil_compression_matrix
            r = np.load(raw_npy, mmap_mode="r")
            cc = coil_compression_matrix(r, a.coil_compress, channels)
            del r
        n_col = info["n_col"]
        n_out = (n_col // 2 + 2) if a.central_only else None      # 146 for 288: the crop the MATLAB video uses
        p = ReconParams(n=n_col, n_spokes=n_spokes, shots=shots, channels=channels,
                        single=not a.double, eps=a.eps, cc_matrix=cc, upsampfac=a.upsampfac, n_out=n_out)
        print(f"[{name}] reconstructing: N={p.n} window={n_spokes} step={shots} "
              f"coils={'all' if channels is None else len(channels)}"
              f"{' ->%d virtual' % a.coil_compress if cc is not None else ''} jobs={a.jobs or os.cpu_count()}")
        t1 = time.time()
        frames = reconstruct(raw_npy, p, out_path=os.path.join(tmp, "frames.npy"), n_jobs=a.jobs, verbose=False)
        t_rec = time.time() - t1
        print(f"[{name}] {frames.shape[0]} frames in {t_rec:.1f}s ({t_rec / frames.shape[0] * 1000:.1f} ms/frame)")

        if a.save_frames:
            np.save(os.path.join(out_dir, name + "_frames.npy"), frames)
        if a.save_mat:
            import hdf5storage
            hdf5storage.savemat(os.path.join(out_dir, name + ".mat"),
                                {"frames": np.ascontiguousarray(np.transpose(frames, (1, 2, 0)))},
                                format="7.3", matlab_compatible=True)
        if not a.no_video:
            from .video import write_video
            t2 = time.time()
            write_video(frames, os.path.join(out_dir, name + ".mp4"), fps=a.spokes_per_sec / shots,
                        auto_scale=not a.no_auto_scale, crf=a.crf, already_cropped=n_out is not None)
            print(f"[{name}] video written in {time.time() - t2:.1f}s")
        del frames                      # release memory maps before the temp folder is deleted
        gc.collect()
    print(f"[{name}] total {time.time() - t0:.1f}s (read {t_read:.1f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="rtMRI radial sliding-window reconstruction")
    ap.add_argument("input", help="folder of .dat files, a single .dat, or a cached (T,coils,N) .npy")
    ap.add_argument("-o", "--out", help="output folder; the mp4 files are written directly into it "
                                       "(default: <input folder>/Processed)")
    ap.add_argument("-j", "--jobs", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--channels", type=int, nargs="*", help="1-based coil subset (default: all)")
    ap.add_argument("--n-spokes", type=int, default=None, help="override window length (default NLin)")
    ap.add_argument("--shots", type=int, default=None, help="override step (default NLin/NSeg)")
    ap.add_argument("--central-only", action="store_true",
                    help="only compute/save the central N/2+2 square (what the video shows); exact, ~3-4x faster")
    ap.add_argument("--double", action="store_true", help="compute in double precision (slower)")
    ap.add_argument("--eps", type=float, default=1e-4, help="NUFFT tolerance (1e-4 is far below MRI noise)")
    ap.add_argument("--upsampfac", type=float, default=1.25, help="finufft upsampling (1.25 = fast, 2.0 = most accurate)")
    ap.add_argument("--coil-compress", type=int, default=0, help="APPROXIMATE: compress to K virtual coils")
    ap.add_argument("--spokes-per-sec", type=float, default=379.94, help="MATLAB's hard-coded rate")
    ap.add_argument("--no-auto-scale", action="store_true",
                    help="video: skip the global brightness normalisation (MATLAB feeds raw magnitudes to imadjust, "
                         "which clips anything >1 to white)")
    ap.add_argument("--save-mat", action="store_true", help="also write MATLAB v7.3 .mat with `frames`")
    ap.add_argument("--save-frames", action="store_true", help="also write <name>_frames.npy")
    ap.add_argument("--crf", type=int, default=23,
                    help="x264 quality: lower = bigger/better (18 ~ visually lossless, 23 ~ MATLAB's default Quality 75, 28 = small)")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args(argv)

    if os.path.isdir(a.input):
        files = sorted(glob.glob(os.path.join(a.input, "*.dat")))
        out = a.out or os.path.join(a.input, "Processed")
    else:
        files = [a.input]
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.input)), "Processed")
    t0 = time.time()
    for f in files:
        process_file(f, out, a)
    print(f"All done: {len(files)} file(s) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
