"""Video output equivalent to the MATLAB VideoWriter block (crop + imadjust + MPEG-4)."""
from __future__ import annotations

import subprocess
import numpy as np


def imadjust_default(img: np.ndarray, tol=(0.01, 0.99), nbins: int = 65536) -> np.ndarray:
    """MATLAB imadjust(I) for a double image: stretchlim([.01 .99]) then linear map to [0, 1].

    MATLAB histograms double images on [0, 1] (values outside are clipped), so this does too.
    """
    x = np.clip(img, 0.0, 1.0)
    hist, _ = np.histogram(x, bins=nbins, range=(0.0, 1.0))
    cdf = np.cumsum(hist) / x.size
    lo = np.searchsorted(cdf, tol[0], side="left") / (nbins - 1)
    hi = np.searchsorted(cdf, tol[1], side="left") / (nbins - 1)
    if hi <= lo:
        lo, hi = 0.0, 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def write_video(frames: np.ndarray, path: str, fps: float, matlab_imadjust: bool = True,
                auto_scale: bool = False, crf: int = 18, already_cropped: bool = False):
    """frames: (T, N, N). Crops the central region exactly as the MATLAB script does."""
    import imageio_ffmpeg
    T, N, _ = frames.shape
    a, b = (0, N) if already_cropped else (N // 4 - 1, N - N // 4 + 1)   # MATLAB dimx/4 : dimx-dimx/4+1
    h = b - a
    scale = 1.0
    if auto_scale:                                       # optional: normalise brightness globally
        scale = 1.0 / float(np.percentile(frames[:: max(1, T // 50), a:b, a:b], 99.9))
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{h}x{h}", "-r", f"{fps:.6f}", "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for t in range(T):
            img = frames[t, a:b, a:b].astype(np.float64) * scale
            img = imadjust_default(img) if matlab_imadjust else np.clip(img, 0, 1)
            proc.stdin.write(np.round(img * 255).astype(np.uint8).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
