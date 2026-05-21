<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Current Design

The repository is intentionally centered on inference. The launcher keeps the upstream flat-import runtime stable, while the surrounding structure makes the actual production path easier to find.

## Runtime Flow

1. `run_demo.py` changes into `src/` and runs `demo.py` with the caller's CLI arguments.
2. `src/demo.py` resolves either one RGB/depth pair or a matched RGB/depth directory, prepares batches, optionally resizes the long side to `--demo_max_size`, and writes outputs.
3. `src/model/infer.py` owns shared model loading, RGB/depth loading, padding/cropping, prediction, and anchor capping.
4. `src/model/ognidc.py` combines the MA-depthmap prior, RGBD backbone, GRU update, multiresolution optimization layer, and SPN refinement.

## Inputs

The model expects an RGB image plus a sparse metric-depth `.npy` with the same image frame. Valid depth values are in meters; invalid pixels are `0`.

Directory inference matches files by stem:

```text
images_2/_DSC8679.JPG
sparse_depth_all_images_2/_DSC8679.npy
```

## Batch And Preview Path

`--demo_batch_size 16 --demo_max_size 512` is the validated high-throughput path for the bicycle sequence. Images are resized with bilinear RGB interpolation. Sparse depth is downsampled by preserving valid anchors with a min-depth style pooling path, so metric constraints survive preview resizing.

## TensorRT Path

`--trt` enables explicit TensorRT engines where they are available:

- MA-depthmap DINOv3-H patch encoder.
- Full 352x512 prior engine.
- Fixed-shape backbone decoder engines.

Missing or failed engines fall back to eager PyTorch. Export helpers live in `tools/export_*_trt.py`.

## CUDA Graph And Final Reps

`--demo_cuda_graph --capturable_inference --cg_fixed_iters <N>` captures fixed-shape inference batches for replay. For validated batch-16 512-preview exposure batches, `src/model/final_reps.py` runs a representative subset of frames and interpolates the final outputs. The selected representative layout depends on endpoint RGB span:

- calibrated span: `(0, 1, 5, 10, 15)`
- generic span: `(0, 3, 6, 10, 15)`
- high span: `(0, 6, 12, 14, 15)`

This is a throughput optimization for sequence processing, not the default for arbitrary one-off images.

## Anchor Cap

The MA prior can extrapolate unconstrained far-field depth beyond the sparse SfM anchors. `apply_anchor_cap` caps predictions above `anchor_cap_factor * max(valid sparse depth)` and writes those pixels as `0`, preserving the same invalid-depth sentinel used by sparse inputs.
