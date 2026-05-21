<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Tools

## Sparse Depth

`generate_colmap_sparse_depth.py` converts a COLMAP sparse reconstruction into full-size sparse depth `.npy` maps:

```powershell
uv run python tools\generate_colmap_sparse_depth.py --model-dir <scene>\sparse\0 --rgb-dir <scene>\images_2 --out-dir <scene>\omnidc_test\sparse_depth_all_images_2
```

By default it uses the more certain COLMAP points only: `--min-track-length 3` and `--max-reproj-error 2`. Use `--no-quality-filter` only for comparison/debugging.

For a one-frame smoke check, add `--only-stem _DSC8679 --limit 1 --verbose`.

## Inference Benchmarks

- `bench_inference.py`: bicycle-focused eager/TRT/final-rep benchmark switches through environment variables.
- `bench_trt.py`: compares eager and TensorRT predictions on the bicycle frame.
- `profile_forward.py`: forward-pass profiling hooks for the model path.

## TensorRT Export

- `export_prior_trt.py`: MA-depthmap patch encoder engine.
- `export_full_prior_512_trt.py`: full 352x512 prior engine.
- `export_backbone_trt.py`: fixed-shape backbone decoder engines.

Engines are written under `checkpoints/trt/`, which is ignored by git.

## DCN

`build_dcn.cmd` rebuilds the deformable convolution extension for the active Python/Torch/CUDA environment. Use it when `import DCN` fails or after changing the local environment.
