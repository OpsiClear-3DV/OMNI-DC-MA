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
- COLMAP sparse-model conversion into per-image sparse metric-depth `.npy` files.

## Current Measured Preview Behavior

For the bicycle batch-16 512-preview path, the retained representative modes measured against the all-frame teacher were:

| Span bucket | Mode | Mean gap | P95 gap | Batch time |
| --- | --- | ---: | ---: | ---: |
| low | `metric_generic16` | 0.009102 m | 0.022091 m | ~0.891 s |
| mid | `hybrid_calibrated16` | 0.008479 m | 0.025354 m | ~0.890 s |
| high | `metric_highspan16` | 0.011834 m | 0.025463 m | ~0.886 s |

The practical recommendation is:

- Use batch-16 512-preview with TensorRT, capturable fixed-iteration CG, CUDA graph replay, and final reps for whole-sequence throughput.
- Use full-resolution batch-1 inference when final per-image fidelity matters more than processing speed.
