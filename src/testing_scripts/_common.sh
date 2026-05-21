#!/usr/bin/env bash
# Shared OMNI-DC-MA inference config. Source this and call
# `run_omnidc_demo <extra demo.py args...>`.
#
#   source "$(dirname "$0")/_common.sh"
#   run_omnidc_demo --demo_rgb a.jpg --demo_depth a.npy --demo_out_dir out/

run_omnidc_demo() {
    local load_dav2=1                 # deprecated no-op; prior (MA depthmap) is always on
    local resolution=3
    local pred_confidence_input=1
    local mrlgw="uniform"             # multi_resolution_learnable_gradients_weights
    local optim_layer_input_clamp=1.0
    local depth_activation_format='exp'
    local max_depth=300.0
    local whiten_sparse_depths=1
    local backbone='rgbd'

    python demo.py \
        --max_depth "$max_depth" --data_normalize_median 1 \
        --num_resolution "$resolution" \
        --multi_resolution_learnable_gradients_weights "$mrlgw" \
        --load_dav2 "$load_dav2" \
        --gpus 0 \
        --GRU_iters 1 \
        --optim_layer_input_clamp "$optim_layer_input_clamp" \
        --depth_activation_format "$depth_activation_format" \
        --whiten_sparse_depths "$whiten_sparse_depths" \
        --gru_internal_whiten_method median \
        --backbone_mode "$backbone" \
        --pred_confidence_input "$pred_confidence_input" \
        "$@"
}
