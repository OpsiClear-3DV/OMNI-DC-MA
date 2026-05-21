<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Optimization Notes

These are the current optimizations carried into this repo.

## Implemented

- Shared inference entry point in `src/model/infer.py`, so demo, benchmarks, and tests use the same load/predict/cap path.
- GPU JPEG decode via `torchvision.io.decode_jpeg` when available, with PIL fallback for unsupported images.
- One-time dummy camera matrix cache instead of rebuilding unused `K` tensors per prediction.
- Single sparse-depth load per pair when output capping is needed.
- Anchor cap to remove unconstrained far-field predictions before saving.
- Jacobi preconditioned CG in `src/model/optim_layer/optim_layer.py`.
- Fixed-iteration capturable CG path for CUDA graph inference.
- TensorRT runtime hooks for the MA-depthmap prior and backbone decoder subgraphs.
- Optional full 352x512 prior TensorRT engine.
- Directory batching, padded batch execution, and output writing from `src/demo.py`.
- Final-output representative interpolation for validated 512-preview batch-16 sequence batches.
- COLMAP sparse-model conversion into per-image sparse metric-depth `.npy` files, using higher-certainty point tracks by default.

## Current Measured Preview Behavior

For the bicycle benchmark, the unmodified OMNI-DC+MA baseline at commit `1e1987b` measured 2.593 s for one image at the original script's padded `1648x2480` resolution. The retained 512-preview representative modes run 16 images at padded `352x512`; the speed gain compares original single-image throughput against preview batch throughput.

| Path | Images/run | Padded resolution | Time/run | Per image | Speed gain | Mean error | P95 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original OMNI-DC+MA (`1e1987b`) | 1 | `1648x2480` | 2.593 s | 2.593 s | 1.0x | n/a | n/a |
| low, `metric_generic16` | 16 | `352x512` | ~0.891 s | ~0.0557 s | ~46.6x | 0.009102 m | 0.022091 m |
| mid, `hybrid_calibrated16` | 16 | `352x512` | ~0.890 s | ~0.0556 s | ~46.6x | 0.008479 m | 0.025354 m |
| high, `metric_highspan16` | 16 | `352x512` | ~0.886 s | ~0.0554 s | ~46.8x | 0.011834 m | 0.025463 m |

The error columns are final-output representative approximation error against the 512-preview all-frame teacher, not ground-truth depth error or a full-resolution original-demo comparison.

The practical recommendation is:

- Use batch-16 512-preview with TensorRT, capturable fixed-iteration CG, CUDA graph replay, and final reps for whole-sequence throughput.
- Use full-resolution batch-1 inference when final per-image fidelity matters more than processing speed.
