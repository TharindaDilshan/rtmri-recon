"""Synthetic multi-coil radial rtMRI data with the same sampling scheme/angles as the MATLAB code."""
import sys, os, numpy as np, finufft
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtmri_recon import radial_trajectory


def make(path, N=288, C=64, S=95, s=5, n_frames=300, seed=0):
    rng = np.random.default_rng(seed)
    T = S + s * (n_frames - 1) + 3
    om1, om2 = radial_trajectory(N, S, s)
    yy, xx = np.mgrid[-1:1:N * 1j, -1:1:N * 1j]
    # coil sensitivities: smooth Gaussians around a ring
    ang = np.linspace(0, 2 * np.pi, C, endpoint=False)
    sens = np.stack([np.exp(-((xx - .8 * np.cos(a)) ** 2 + (yy - .8 * np.sin(a)) ** 2) / 0.8) *
                     np.exp(1j * (a + xx * np.sin(a))) for a in ang]).astype(np.complex64)
    raw = np.lib.format.open_memmap(path, mode="w+", dtype=np.complex64, shape=(T, C, N))
    n_groups = int(np.ceil(T / s))
    for g in range(n_groups):
        t = g / 20.0
        img = ((xx / .6) ** 2 + (yy / .8) ** 2 < 1) * 1.0
        img += ((xx - .2 * np.sin(t)) ** 2 + (yy + .3 - .15 * np.cos(t)) ** 2 < .04) * 1.5
        img += ((xx + .3) ** 2 + (yy - .1) ** 2 < .02) * .7
        idx = np.arange(g * s, min((g + 1) * s, T))
        slots = idx % S
        coil_imgs = (sens * img[None]).astype(np.complex64)
        # forward NUFFT (type 2) at these spokes, same convention as the adjoint used in recon
        d = finufft.nufft2d2(om1[slots].ravel().astype(np.float32), om2[slots].ravel().astype(np.float32),
                             coil_imgs, isign=-1, eps=1e-5)          # (C, len*N)
        d = d.reshape(C, len(idx), N).transpose(1, 0, 2)
        d += (rng.standard_normal(d.shape) + 1j * rng.standard_normal(d.shape)).astype(np.complex64) * 0.01 * np.abs(d).max()
        raw[idx] = d
    raw.flush()
    return T


if __name__ == "__main__":
    out = sys.argv[1]; nf = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    print("T =", make(out, n_frames=nf))
