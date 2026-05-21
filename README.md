<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# OMNI-DC-MA

**TL;DR:** OMNI-DC-MA is an inference-focused depth-completion repo for turning RGB images plus sparse COLMAP/SfM depth anchors into dense metric depth maps. It is a cleaned-up, Windows/CUDA-13-ready fork of the current OMNI-DC inference path with a Metric-Anything depth prior, higher-certainty COLMAP anchor generation, TensorRT hooks, batch processing, and release-hosted model assets.

<p align="center">
  <img src="docs/assets/bicycle_sparse_vs_completed.png" alt="Sparse bicycle COLMAP/SfM depth anchors projected into the image plane next to the completed OMNI-DC-MA depth map" width="100%">
</p>

<p align="center"><em>Example bicycle frame: globally filtered sparse metric SfM anchors projected to 2D as the input depth signal, compared with the regenerated 512 px OMNI-DC-MA depth map.</em></p>

## What This Repo Is For

Use this repo when you have a COLMAP-style reconstruction or sparse metric depth maps and want dense per-image depth for downstream 3D work, such as 3DGS preprocessing, point cloud generation, or scene inspection. The repo keeps only the inference surface and the support tools needed to run it cleanly.

The expected input is:

- RGB images, usually from a scene directory such as `images_2/`.
- Sparse metric depth `.npy` files with matching stems, where `0` means invalid.

The output is:

- Dense metric depth `.npy` files.
- Optional depth visualizations as `.png`.
- Optional raw and sky/far-field masks.

## What Is Better Than The Original Method

Compared with the original research repo / earlier local pipeline, this version is meant to be easier to run and better suited for whole-scene inference:

- **Inference-first layout:** training datasets, losses, experiment folders, and generated outputs are not part of the repo surface.
- **Repo-root launcher:** `python run_demo.py ...` works from the root instead of requiring manual `cd src` import setup.
- **Metric-Anything prior:** replaces the older monocular prior path with an in-tree MA-depthmap prior wrapper.
- **Higher-certainty sparse anchors:** the COLMAP converter defaults to `track_length >= 3` and `reprojection_error <= 2 px`, and can optionally reject anchors that disagree with a reference depth map in inverse depth.
- **Batch directory processing:** basename-matched RGB/depth directories can be processed in one command.
- **Fast 512 px preview path:** batch-16 preview inference supports TensorRT, fixed-iteration CG, CUDA graph replay, and final-output representative interpolation, giving about 24x higher per-image throughput than the original single-image OMNI-DC+MA path at the same 512-preview image size.
- **Safer saved output:** anchor capping zeros unconstrained far-field predictions beyond `anchor_cap_factor * max(valid sparse depth)`.
- **Release-hosted model assets:** large weights and optional native extension binaries are GitHub release assets, not git-tracked files.
- **Smoke tests and docs:** import tests, tool docs, design notes, and optimization notes are included.

## Speed vs Original

The original OMNI-DC+MA path is a single-image research demo. It loads one RGB/depth pair, runs eager PyTorch, solves CG to convergence, and writes one result. OMNI-DC-MA keeps the same core depth-completion model but adds a throughput path for scene sweeps:

- Batch-16 directory inference instead of one image at a time.
- 512 px preview resizing for fast scene-wide depth proposals.
- TensorRT engines for the MA-depthmap prior and backbone decoder subgraphs.
- Fixed-iteration capturable CG (`--cg_fixed_iters 120`) instead of variable convergence loops.
- CUDA graph replay to remove repeated Python/allocator overhead at fixed shapes.
- Final-output representative interpolation for validated 512-preview exposure batches.

The baseline below is the unmodified OMNI-DC+MA commit (`1e1987b`) run as a single-image path on the bicycle frame with the same higher-certainty sparse depth input and the same 512-long-side image size: **1.337 s/image** steady-state, resized to `340x512` and padded to `352x512`, with 9.21 GB peak VRAM. The retained OMNI-DC-MA preview modes run 16 images at the same padded `352x512` shape in about **0.89 s total**, or **0.055-0.056 s/image**, so serially processing the same 16 preview-sized images with the original path would take about **21.4 s**.

| Path | Images/run | Padded resolution | Time/run | Per image | Speed gain vs original OMNI-DC+MA | Mean error vs 512 all-frame teacher | P95 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original OMNI-DC+MA (`1e1987b`) | 1 | `352x512` | 1.337 s | 1.337 s | 1.0x | n/a | n/a |
| OMNI-DC-MA low span, `metric_generic16` | 16 | `352x512` | ~0.891 s | ~0.0557 s | ~24.0x | 0.009102 m | 0.022091 m |
| OMNI-DC-MA mid span, `hybrid_calibrated16` | 16 | `352x512` | ~0.890 s | ~0.0556 s | ~24.0x | 0.008479 m | 0.025354 m |
| OMNI-DC-MA high span, `metric_highspan16` | 16 | `352x512` | ~0.886 s | ~0.0554 s | ~24.1x | 0.011834 m | 0.025463 m |

The speed gain compares original single-image throughput against OMNI-DC-MA preview throughput at the same padded image size. The error columns are the final-output representative approximation error against the 512-preview all-frame teacher, not ground-truth depth error.

That path is meant for fast scene processing and preview-quality dense depth. For final per-image fidelity, use full-resolution batch-1 inference.

## Repository Map

| Path | Purpose |
| --- | --- |
| `run_demo.py` | Repo-root launcher for single-image and directory inference. |
| `src/demo.py` | Batching, 512 px resizing, CUDA graph handling, and output writing. |
| `src/model/` | OGNIDC, MA-depthmap prior, optimization layer, TensorRT hooks, final-rep interpolation. |
| `tools/generate_colmap_sparse_depth.py` | Converts COLMAP sparse models into per-image sparse depth `.npy` inputs. |
| `tools/build_trt_engines.ps1` | Local TensorRT engine build wrapper for the retained prior and decoder engines. |
| `tools/export_*_trt.py` | TensorRT export helpers. |
| `tests/` | Import smoke tests and optional local bicycle regression. |
| `docs/` | Current design, optimization notes, release notes, and README image asset. |

## Setup

```powershell
git clone https://github.com/OpsiClear-3DV/OMNI-DC-MA.git
cd OMNI-DC-MA
uv sync --extra trt
```

If the prebuilt DCN extension is unavailable or incompatible with your machine, rebuild it:

```powershell
tools\build_dcn.cmd
```

## Model Assets

The repo loads OMNI-DC weights from HuggingFace by default. Pinned assets are also published on the GitHub release:

- `omnidc_v1.1.safetensors`: OMNI-DC weights.
- `metricanything_student_depthmap.pt`: Metric-Anything depth-prior weights.
- `DCN.cp312-win_amd64.pyd`: optional prebuilt Windows x64 extension for Python 3.12, Torch 2.11.0+cu130, CUDA 13.0, Blackwell `sm_120`.
- `SHA256SUMS.txt`: integrity checksums.

```powershell
gh release download v0.1.0 -R OpsiClear-3DV/OMNI-DC-MA --dir release_assets
```

TensorRT engines are intentionally not shipped in git or the release because they are tied to the local GPU, TensorRT version, CUDA version, driver, and fixed input shapes. Build them on the target machine:

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_trt_engines.ps1
```

The default build creates the dynamic MA-depthmap patch-prior engine, the fixed 352x512 prior engine, and the retained batch-16/batch-5 512-preview decoder engines under `checkpoints\trt`. Add `-IncludeFullResolutionBackbone` to also build the full-resolution batch-1 decoder engines, or `-DryRun` to print the commands without building.

Large model files, TensorRT engines, datasets, predictions, and visualizations are ignored by git.

## Generate Sparse Depth From COLMAP

Convert a COLMAP sparse model into OMNI-DC sparse depth maps:

```powershell
uv run python tools\generate_colmap_sparse_depth.py `
  --model-dir C:\path\to\scene\sparse\0 `
  --rgb-dir C:\path\to\scene\images_2 `
  --out-dir C:\path\to\scene\omnidc_test\sparse_depth_all_images_2_certain
```

Default point filtering keeps the more certain COLMAP tracks:

- `--min-track-length 3`
- `--max-reproj-error 2`

To also reject SfM anchors that disagree too much with an existing dense depth map, add a matched `.npy` depth directory and an inverse-depth threshold:

```powershell
uv run python tools\generate_colmap_sparse_depth.py `
  --model-dir C:\path\to\scene\sparse\0 `
  --rgb-dir C:\path\to\scene\images_2 `
  --out-dir C:\path\to\scene\omnidc_test\sparse_depth_all_images_2_certain_consistent `
  --reference-depth-dir C:\path\to\scene\omnidc_test\pred_current_all_images_512_certain `
  --max-relative-inverse-depth-error 0.25 `
  --drop-inconsistent-points
```

Use `--max-inverse-depth-error` for an absolute inverse-depth threshold in `1/m`. Add `--align-reference-depth-scale` when the reference maps are only relatively scaled.
By default, only the failing point observation is removed; add `--drop-inconsistent-points` to remove that COLMAP 3D point from every selected output view if any observation fails.

The checked-in bicycle example was regenerated with `--max-relative-inverse-depth-error 0.25 --drop-inconsistent-points`, using `pred_current_all_images_512_certain` as the reference. It wrote `sparse_depth_all_images_2_certain_ref25_global` with 194 maps, 219 globally rejected COLMAP points, and 117,331 sparse anchor pixels.

Use `--no-quality-filter` only for comparison/debugging.

## Run Completion

Single image:

```powershell
uv run python run_demo.py `
  --gpus 0 `
  --load_dav2 1 --num_resolution 3 `
  --multi_resolution_learnable_gradients_weights uniform `
  --GRU_iters 1 --optim_layer_input_clamp 1.0 `
  --depth_activation_format exp --whiten_sparse_depths 1 `
  --gru_internal_whiten_method median --backbone_mode rgbd `
  --pred_confidence_input 1 --max_depth 300.0 --data_normalize_median 1 `
  --demo_rgb C:\path\to\image.jpg `
  --demo_depth C:\path\to\sparse_depth.npy `
  --demo_out_dir outputs\single `
  --demo_outputs depth,raw,vis `
  --trt --anchor_cap_factor 2
```

Whole-scene 512 px preview path:

```powershell
uv run python run_demo.py `
  --gpus 0 `
  --load_dav2 1 --num_resolution 3 `
  --multi_resolution_learnable_gradients_weights uniform `
  --GRU_iters 1 --optim_layer_input_clamp 1.0 `
  --depth_activation_format exp --whiten_sparse_depths 1 `
  --gru_internal_whiten_method median --backbone_mode rgbd `
  --pred_confidence_input 1 --max_depth 300.0 --data_normalize_median 1 `
  --demo_rgb_dir C:\path\to\scene\images_2 `
  --demo_depth_dir C:\path\to\scene\omnidc_test\sparse_depth_all_images_2_certain_ref25_global `
  --demo_out_dir C:\path\to\scene\omnidc_test\pred_current_all_images_512_certain_ref25_global `
  --demo_batch_size 16 --demo_max_size 512 `
  --demo_outputs depth,vis `
  --trt --capturable_inference --cg_fixed_iters 120 --demo_cuda_graph `
  --anchor_cap_factor 2
```

For maximum per-image fidelity, use full resolution and batch 1. For throughput on scene sweeps, use the 512 px batch path.

## Output Files

For each RGB stem:

- `<stem>.npy`: capped dense metric depth.
- `<stem>_raw.npy`: raw dense depth when it differs from capped output.
- `<stem>.png`: color depth visualization.
- `<image-name>.png`: optional sky/far-field mask when `skymask` is requested.

`anchor_cap_factor` defaults to `2`, which zeros predictions farther than twice the deepest valid sparse anchor. This keeps the output compatible with the sparse-depth convention that `0` means invalid.

## Verification

```powershell
uv run ruff check run_demo.py src\demo.py src\config.py src\model\infer.py src\model\final_reps.py tools tests
uv run pytest tests\test_imports.py
```

The optional bicycle regression test is gated on local CUDA, weights, and the local bicycle dataset path.

## More Detail

- [Current design](docs/current-design.md)
- [Optimization notes](docs/optimization-notes.md)
- [Tools guide](tools/README.md)
- [v0.1.0 release notes](docs/release-v0.1.0.md)

## Licensing And Attribution

OMNI-DC-MA is a mixed-license repository, not a single-license AGPL relicensing of every file.

New OMNI-DC-MA content is licensed under AGPL-3.0-only; see `LICENSE` for the AGPL text. Upstream and vendored components retain their original licenses, copied under `LICENSES/` and mapped in `NOTICE.md`.

This repository includes code derived from Princeton `princeton-vl/OMNI-DC`, Metric-Anything, DINOv3, DCNv2, COLMAP utilities, and RAFT-adapted update blocks. This product is built with DINOv3.
