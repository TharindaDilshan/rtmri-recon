import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtmri_recon import ReconParams, reconstruct, reconstruct_reference

if __name__ == "__main__":
    path = sys.argv[1]
    raw = np.load(path, mmap_mode="r")
    N, S, s = raw.shape[2], 95, 5
    sub = np.asarray(raw[: S + s * 11 + 3])                      # 12 frames for the slow baseline
    p64 = ReconParams(n=N, n_spokes=S, shots=s, single=False, eps=1e-9)
    t = time.time(); ref = reconstruct_reference(sub, p64); tb = (time.time() - t) / ref.shape[0]
    print(f"literal MATLAB-style loop (finufft, double, all coils regridded per frame): {tb*1000:.0f} ms/frame")
    for label, kw, jobs in [("incremental double 1 job", dict(single=False, eps=1e-9), 1),
                            ("incremental single 1 job", dict(single=True), 1),
                            ("incremental single 2 jobs", dict(single=True), 2),
                            ("single, upsampfac=1.25, 2 jobs", dict(single=True, upsampfac=1.25), 2)]:
        p = ReconParams(n=N, n_spokes=S, shots=s, **kw)
        t = time.time(); out = np.array(reconstruct(sub, p, n_jobs=jobs, verbose=False)); dt = (time.time() - t) / out.shape[0]
        print(f"{label:34s}: {dt*1000:6.0f} ms/frame  speedup x{tb/dt:4.1f}  maxrelerr {np.abs(out-ref).max()/ref.max():.1e}")
