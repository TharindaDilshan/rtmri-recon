import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtmri_recon import ReconParams, reconstruct
from rtmri_recon.coilcomp import coil_compression_matrix
if __name__ == "__main__":
    raw = np.load(sys.argv[1], mmap_mode="r"); jobs = int(sys.argv[2])
    N, S, s = raw.shape[2], 95, 5
    nF = (raw.shape[0] - S + s - 1) // s
    ref = None
    cc16 = coil_compression_matrix(raw, 16)
    cfgs = [("double, upsamp 2, eps1e-9, full FOV", dict(single=False, eps=1e-9, upsampfac=2.0)),
            ("single, upsamp 2, eps1e-5, full FOV", dict(single=True, eps=1e-5, upsampfac=2.0)),
            ("single, upsamp 1.25, eps1e-4, full FOV", dict(single=True, eps=1e-4, upsampfac=1.25)),
            ("... + central 146x146 only", dict(single=True, eps=1e-4, upsampfac=1.25, n_out=146)),
            ("... + 16 virtual coils (approx.)", dict(single=True, eps=1e-4, upsampfac=1.25, n_out=146, cc_matrix=cc16))]
    for label, kw in cfgs:
        p = ReconParams(n=N, n_spokes=S, shots=s, **kw)
        t = time.time(); out = np.array(reconstruct(sys.argv[1], p, n_jobs=jobs, verbose=False)); dt = time.time() - t
        if ref is None: ref = out; refc = out[:, 71:217, 71:217]
        o = out if out.shape[1] == ref.shape[1] else out
        r = ref if out.shape[1] == ref.shape[1] else refc
        print(f"{label:42s}: {dt:6.1f}s total  {dt/nF*1000:6.1f} ms/frame  rel.err vs double {np.abs(o-r).max()/r.max():.1e}  (rms {np.sqrt(((o-r)**2).mean())/np.sqrt((r**2).mean()):.1e})", flush=True)
