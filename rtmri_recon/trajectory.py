"""Radial trajectory, spoke ordering and density compensation.

Line-by-line equivalents of the corresponding blocks of RTMRIProcess.m
(0-based arrays; MATLAB 1-based indices are noted in comments).
"""
from __future__ import annotations

import numpy as np


def spoke_angles(n_spokes: int, shots: float) -> np.ndarray:
    """Return phi[j] (radians, in [0, 2*pi)) for each spoke slot j = 0..n_spokes-1.

    MATLAB:
        ang(j) = 0 - (ceil(j/shots)-1)*seg_ang - shot_ang*((j-1) - shots*(ceil(j/shots)-1))
        phi = mod(-ang/180*pi, 2*pi)
    """
    j = np.arange(1, n_spokes + 1, dtype=np.float64)          # MATLAB j = 1..n_spokes
    seg_ang = 360.0 / n_spokes
    shot_ang = 360.0 / shots
    c = np.ceil(j / shots) - 1.0
    ang = 0.0 - c * seg_ang - shot_ang * ((j - 1.0) - shots * c)
    phi = -(ang / 180.0 * np.pi)
    return np.mod(phi, 2.0 * np.pi)


def radial_trajectory(n: int, n_spokes: int, shots: float):
    """k-space sample positions in radians/sample, shape (n_spokes, n) each.

    The MATLAB code builds rho = -144..143 (/288) so that the DC sample sits at
    (1-based) index 145, i.e. 0-based index n//2.  Returns (om1, om2) with
    om1 = 2*pi*Re(k), om2 = 2*pi*Im(k), k = rho * exp(-1j*phi).
    """
    phi = spoke_angles(n_spokes, shots)
    rho = (np.arange(n) - n // 2) / n                          # -0.5 .. 0.5, DC at index n//2
    k = rho[None, :] * np.exp(-1j * phi)[:, None]              # (n_spokes, n)
    return 2.0 * np.pi * k.real, 2.0 * np.pi * k.imag


def radial_dcf(n: int, n_spokes: int) -> np.ndarray:
    """Density compensation along a spoke, shape (n,).

    MATLAB: dcfRow(i) = |N/2 - (i-0.5)|, i = 1..N, then * pi/n_spokes.
    (Note: this ramp is centred between samples N/2-1 and N/2 whereas the
    trajectory has its DC sample at N/2; kept as-is for fidelity.)
    """
    i = np.arange(1, n + 1, dtype=np.float64)
    return np.pi / n_spokes * np.abs(n / 2.0 - (i - 0.5))
