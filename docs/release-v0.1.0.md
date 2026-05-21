<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# OMNI-DC-MA v0.1.0

Initial organized inference-focused release.

## Included Release Assets

| File | Purpose |
| --- | --- |
| `omnidc_v1.1.safetensors` | OMNI-DC model weights mirrored from `zuoym15/OMNI-DC`. |
| `metricanything_student_depthmap.pt` | MA-depthmap prior weights mirrored from `yjh001/metricanything_student_depthmap`. |
| `DCN.cp312-win_amd64.pyd` | Optional prebuilt Windows x64 extension for Python 3.12, Torch 2.11.0+cu130, CUDA 13.0, Blackwell `sm_120`. |
| `SHA256SUMS.txt` | Checksums for the release assets. |

## Download

```powershell
gh release download v0.1.0 -R OpsiClear-3DV/OMNI-DC-MA --dir release_assets
```

Use the `.safetensors` / `.pt` files for pinned offline inference, or keep the default HuggingFace runtime download path.
