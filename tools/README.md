<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Tools

## Sparse Depth

`generate_colmap_sparse_depth.py` converts a COLMAP sparse reconstruction into full-size sparse depth `.npy` maps:

```powershell
uv run python tools\generate_colmap_sparse_depth.py --model-dir <scene>\sparse\0 --rgb-dir <scene>\images_2 --out-dir <scene>\omnidc_test\sparse_depth_all_images_2_certain
```

By default it uses the more certain COLMAP points only: `--min-track-length 3` and `--max-reproj-error 2`. Use `--no-quality-filter` only for comparison/debugging.

The CLI is a thin wrapper over `src/colmap_utils/`, so the same filters and projection logic can be reused by future COLMAP model editing tools without duplicating script code.

To reject points that disagree with an existing dense depth map, pass matched reference `.npy` maps and an inverse-depth threshold:

```powershell
uv run python tools\generate_colmap_sparse_depth.py --model-dir <scene>\sparse\0 --rgb-dir <scene>\images_2 --out-dir <scene>\omnidc_test\sparse_depth_consistent --reference-depth-dir <scene>\omnidc_test\pred_512 --max-relative-inverse-depth-error 0.25
```

`--max-relative-inverse-depth-error` is a symmetric relative inverse-depth check. `--max-inverse-depth-error` is an absolute inverse-depth check in `1/m`. Add `--align-reference-depth-scale` if the reference depth maps need per-image median scale alignment before filtering. By default only the failing observation is removed; add `--drop-inconsistent-points` to remove that COLMAP 3D point from every selected output view.

For a one-frame smoke check, add `--only-stem _DSC8679 --limit 1 --verbose`.

## COLMAP Utility Modules

Reusable helpers live in `src/colmap_utils/`:

- `filters.py`: quality and inverse-depth consistency filters.
- `sparse_depth.py`: sparse-depth generation from COLMAP tracks.
- `editing.py`: model-editing primitives for removing observations, removing points, adding tracked points, and validating point references.
- `io.py`: thin wrappers around the vendored COLMAP read/write implementation.

The current sparse-depth CLI filters generated `.npy` anchors. The editing helpers are the base for tools that rewrite `images.bin` and `points3D.bin`.

## Inference Benchmarks

- `bench_inference.py`: bicycle-focused eager/TRT/final-rep benchmark switches through environment variables.
- `bench_trt.py`: compares eager and TensorRT predictions on the bicycle frame.
- `profile_forward.py`: forward-pass profiling hooks for the model path.

## TensorRT Export

- `build_trt_engines.ps1`: one-command local build wrapper for the runtime TensorRT engines.
- `export_prior_trt.py`: MA-depthmap patch encoder engine.
- `export_full_prior_512_trt.py`: full 352x512 prior engine.
- `export_backbone_trt.py`: fixed-shape backbone decoder engines.

Build the retained 512-preview engine set from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_trt_engines.ps1
```

The default script builds:

- `prior_dinov3h_fp16.engine`
- `prior_full_352x512_fp16.engine`
- batch-16 and batch-5 `dec6` through `dec3` decoder engines for the 512-preview path.

Useful switches:

- `-IncludeFullResolutionBackbone`: also build batch-1 full-resolution `dec6` through `dec2` engines.
- `-SkipPatchPrior`: skip the large dynamic patch-prior engine when you only need the fixed 352x512 path.
- `-Stage engine`: rebuild engines from existing ONNX files.
- `-DryRun`: print the export commands without running them.

Engines are written under `checkpoints/trt/`, which is ignored by git. They are not release assets because TensorRT engines are specific to the local GPU, CUDA, TensorRT, driver, and exported input shapes.

## DCN

`build_dcn.cmd` rebuilds the deformable convolution extension for the active Python/Torch/CUDA environment. Use it when `import DCN` fails or after changing the local environment.
