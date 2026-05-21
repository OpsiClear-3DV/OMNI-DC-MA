#!/usr/bin/env bash
# OMNI-DC-MA on a single Mip-NeRF 360 bicycle frame (SfM-derived sparse depth).
# Run from src/:   uv run --project .. bash testing_scripts/bicycle_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source testing_scripts/_common.sh

BICYCLE=/c/Users/opsiclear/Desktop/Data_WS1/360_v2/bicycle

run_omnidc_demo \
    --demo_rgb     "$BICYCLE/images_2/_DSC8679.JPG" \
    --demo_depth   "$BICYCLE/omnidc_test/sparse_depth_all_images_2/_DSC8679.npy" \
    --demo_out_dir "$BICYCLE/omnidc_test/pred"
