<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Attribution And License Notice

OMNI-DC-MA is a mixed-license repository, not a single-license AGPL relicensing of every file. New repository organization, documentation, and helper tooling added for OMNI-DC-MA are licensed under the GNU Affero General Public License v3.0 only; see `LICENSE` and `LICENSES/AGPL-3.0-only.txt`.

The AGPL notice for new OMNI-DC-MA content does not remove or replace upstream licenses for copied, vendored, or adapted components. Those components retain their original notices and license terms.

## New OMNI-DC-MA Content

Unless a file or path is listed under a third-party/upstream component below, new content added for this organized repo is licensed as:

```text
SPDX-License-Identifier: AGPL-3.0-only
Copyright (C) 2026 OMNI-DC-MA contributors
```

This includes the repo-level README/docs, project metadata, the sparse-depth generation wrapper, and the licensing/attribution files themselves.

## Upstream And Vendor Components

| Path | Source / attribution | License file |
| --- | --- | --- |
| `src/config.py`, `src/demo.py`, `src/model/` except noted subtrees, `src/testing_scripts/`, inherited tests/tools | Derived from OMNI-DC / the local OMNI-DC inference fork, originally based on Princeton `princeton-vl/OMNI-DC` by Yiming Zuo, Willow Yang, Zeyu Ma, and Jia Deng. | `LICENSES/OMNI-DC-BSD-3-Clause.txt` |
| `src/model/ma_depthmap/` except `network/dinov3/` | Vendored Metric-Anything Prompt-Free Depth Map student from `metric-anything/metric-anything`. | `LICENSES/METRIC-ANYTHING-APACHE-2.0.txt` |
| `src/model/ma_depthmap/network/dinov3/` | Meta DINOv3 code. This product is built with DINOv3. | `LICENSES/DINOV3-LICENSE.md` |
| `src/model/deformconv/` | DCNv2 deformable-convolution extension from `CharlesShang/DCNv2`, modernized locally for the current runtime. | `LICENSES/DCNV2-BSD-3-Clause.txt` |
| `src/model/deformconv/src/cuda/deform_psroi_pooling_cuda.cu` | Microsoft-origin deformable PSROI pooling implementation as indicated in the file header. | `LICENSES/MICROSOFT-MIT.txt` |
| `src/robust_dc_protocol/read_write_colmap_model.py` | COLMAP sparse model reader/writer utility, copyright ETH Zurich and UNC Chapel Hill. | `LICENSES/COLMAP-READER-BSD-3-Clause.txt` |
| `src/model/convgru.py` | Notes adaptation from Princeton RAFT update-block code. | `LICENSES/RAFT-BSD-3-Clause.txt` |

Weights, datasets, TensorRT engines, generated sparse-depth maps, and predictions are not included in this repository. Their upstream or provider terms apply when downloaded or generated separately.
