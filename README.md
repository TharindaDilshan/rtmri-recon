# rtmri-recon

Fast, CPU-only reconstruction of real-time radial MRI (rtMRI) data acquired on Siemens scanners.
It reads raw TWIX (`.dat`) files and writes one MP4 movie per scan, using all available CPU cores.
No GPU is required.

## What it does

For each `.dat` file the pipeline:

1. reads the raw multi-coil radial k-space (via [pymapVBVD](https://github.com/wtclarke/pymapvbvd));
2. builds the radial trajectory and density compensation from the acquisition header;
3. reconstructs a **sliding-window** series of frames: each frame uses a window of `NLin` spokes,
   and the window advances by `NLin / NSeg` spokes per frame;
4. applies a gridding (non-uniform FFT) reconstruction per coil and combines coils by
   root-sum-of-squares;
5. crops to the region of interest, applies a per-frame contrast stretch and writes an H.264 MP4.

The video frame rate is `spokes_per_second / (NLin / NSeg)` (default 379.94 spokes/s, i.e. 20 fps for a
19-spoke step).

## Installation

Requires Python 3.9 or newer.

```
pip install -r requirements.txt
```

Dependencies: `numpy`, `finufft`, `pymapvbvd`, `imageio-ffmpeg` (bundles an ffmpeg binary, nothing else to
install), and optionally `hdf5storage` (only for `--save-mat`).

## Usage

Run from the repository folder (the folder that contains `rtmri_recon/`):

```
python -m rtmri_recon <input> [options]
```

`<input>` is a folder of `.dat` files (every `*.dat` in it is processed) or a single `.dat` file.

Example:

```
python -m rtmri_recon C:\rtMRI\data\TWIX -o C:\rtMRI\data\unaligned --central-only --crf 18
```

This processes every `.dat` in `C:\rtMRI\data\TWIX` and writes `<scan name>.mp4` for each one directly
into `C:\rtMRI\data\unaligned` (no sub-folders are created). Without `-o`, the movies go into a
`Processed` folder inside the input folder.

Only the MP4 files are written. Intermediate data lives in a temporary folder that is deleted
automatically; your `.dat` files are never copied or modified.

### Options

| Option | Default | Meaning |
|---|---|---|
| `-o, --out DIR` | `<input>/Processed` | Output folder; MP4s are written directly into it. |
| `--central-only` | off | Reconstruct only the central `N/2+2` square (146×146 for N=288), which is the region shown in the video. Same pixel values as the full image in that region, roughly 3–4× faster. |
| `--crf N` | 23 | H.264 quality. Lower = larger, better quality (18 ≈ visually lossless, 23 = good, 28 = small). |
| `-j, --jobs N` | all cores | Number of worker processes. |
| `--channels 1 2 5 ...` | all coils | 1-based coil subset to use. |
| `--spokes-per-sec X` | 379.94 | Acquisition rate; sets the video frame rate. |
| `--no-auto-scale` | off | Skip the global brightness normalisation applied before the per-frame contrast stretch. |
| `--no-video` | off | Skip MP4 output (useful with `--save-frames` / `--save-mat`). |
| `--save-frames` | off | Also save the reconstructed frames as `<name>_frames.npy` (float32, frames × x × y). |
| `--save-mat` | off | Also save a MAT-file (v7.3) with `frames` (x, y, t). Needs `hdf5storage`. |
| `--double` | off | Compute in double precision (slower, not needed for normal use). |
| `--eps X` | 1e-4 | NUFFT accuracy. 1e-4 is far below MRI noise. |
| `--upsampfac X` | 1.25 | NUFFT oversampling (1.25 fast; 2.0 most accurate). |
| `--n-spokes`, `--shots` | from header | Override window length (`NLin`) and step (`NLin/NSeg`). Required when the input is a pre-extracted `.npy`. |
| `--coil-compress K` | off | **Approximate**: compress to K virtual coils for extra speed. Check image quality on your data before using. |

On Windows the command must be run as a module (`python -m rtmri_recon ...`) from a normal command
prompt or script, as above.

## Performance

Reconstruction typically takes a few milliseconds to a few tens of milliseconds per frame, depending on
core count and options. For example, a 595-frame scan (288×288 base resolution, 64 coils) reconstructs in
about 6–7 s on a 24-core machine with `--central-only`. Reading the `.dat` file and encoding the video add
a few seconds.

How it is fast:

* **Incremental window update.** Gridding is linear, and only `NLin/NSeg` spokes change between frames,
  so each frame is the previous per-coil images plus the gridded *change* in k-space, instead of
  re-gridding every spoke of every coil. An exact re-computation is done periodically (every 256 frames)
  to bound floating-point drift.
* **Parallel frames.** The frame range is split into chunks processed by separate processes; each chunk
  initialises itself exactly, so results do not depend on the number of workers.
* **finufft** (multithreaded CPU NUFFT) in single precision.
* **Central region only** (`--central-only`): image pixels are independent outputs of the transform, so
  computing just the displayed region is exact and cheaper.
* The raw data is streamed through a spoke-major on-disk cache, so a scan never has to fit in RAM.

## Reconstruction details

* Trajectory: radial spokes in an interleaved order defined by `NLin` and `NSeg`; the DC
  sample is at index `N/2`.
* Density compensation: linear ramp `|N/2 − (i − 0.5)|` scaled by `π / NLin`.
* Each spoke that enters the window is scaled by 2 relative to the initial window (`ReconParams.new_gain`).
* Image: `sqrt(Σ_coils |x_c|²)` where `x_c` is the adjoint NUFFT of the density-compensated data,
  normalised by `N`.
* Video: central crop, then each frame is stretched between its 1st and 99th intensity percentiles. Frames
  are first scaled so the 99.9th percentile of the whole movie is 1.0 (otherwise values above 1 would
  saturate); use `--no-auto-scale` to disable this.

## Project layout

```
rtmri_recon/
  __main__.py     entry point (python -m rtmri_recon)
  pipeline.py     command line, per-file driver
  reader.py       TWIX -> on-disk spoke array
  trajectory.py   spoke angles, radial trajectory, density compensation
  recon.py        sliding-window reconstruction (parallel, incremental)
  video.py        crop, contrast stretch, MP4 writer
  coilcomp.py     optional SVD coil compression
tests/
  test_equivalence.py   correctness tests
  make_synthetic.py     synthetic multi-coil radial data generator
  benchmark.py, benchmark2.py   timing scripts
```

## Testing

```
python tests/test_equivalence.py
```

checks that (a) the NUFFT matches a brute-force evaluation of the underlying Fourier sum, and (b) the
incremental, parallel reconstruction equals a straightforward frame-by-frame reference, including the
trailing-spoke edge case, coil subsets and central cropping.

To benchmark on synthetic data:

```
python tests/make_synthetic.py synth_raw.npy 300
python tests/benchmark2.py synth_raw.npy 4
```

## Troubleshooting

* **Out of memory / slow disk:** raw data is cached on disk in the system temp folder; make sure it has
  free space about the size of the raw acquisition (set `TMP`/`TEMP` to another drive if needed).
* **Fewer cores wanted:** use `-j 4`.
* **Video looks too bright or too dark:** try `--no-auto-scale`, or adjust `--crf` for compression only.
* **Different crop or step than the header implies:** use `--n-spokes` and `--shots`.
