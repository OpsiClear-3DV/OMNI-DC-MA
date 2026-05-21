import sys
from pathlib import Path

# Sibling modules (backbone, convgru, ...) are imported by bare name; make this
# directory importable before those imports run.
sys.path.append(str(Path(__file__).parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from backbone import Backbone  # noqa: E402
from convgru import BasicUpdateBlock  # noqa: E402
from huggingface_hub import PyTorchModelHubMixin  # noqa: E402

# Repo-owned depth prior: vendored copy of Metric-Anything's Prompt-Free
# Depth Map student (replaces the original DAv2 ViT-L prior).
from ma_depthmap import MADepthMapPrior  # noqa: E402
from optim_layer.optim_layer import DepthGradOptimLayer, set_jacobi  # noqa: E402
from tensor_stats import quantile_02_98_flat  # noqa: E402


def upsample_depth(depth, mask, r=8):
    """ Upsample depth field [H/r, W/r, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = depth.shape  # B x 1 x H x W
    mask = mask.view(N, 1, 9, r, r, H, W)
    mask = torch.softmax(mask, dim=2)

    up_depth = F.unfold(depth, [3, 3], padding=1)
    up_depth = up_depth.view(N, 1, 9, 1, 1, H, W)

    up_depth = torch.sum(mask * up_depth, dim=2)
    up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
    return up_depth.reshape(N, 1, r * H, r * W)

def _median_nonzero_flat(flat: torch.Tensor, default: float = 1.0) -> torch.Tensor:
    """Per-row median over positive values with no tensor-to-Python branch."""
    valid = flat > 0
    counts = valid.sum(dim=1)
    sorted_valid = flat.masked_fill(~valid, float("inf")).sort(dim=1).values
    idx = ((counts.clamp(min=1) - 1) // 2).long()
    med = sorted_valid.gather(1, idx[:, None]).squeeze(1)
    return torch.where(counts > 0, med, torch.full_like(med, default))


class OGNIDC(nn.Module, PyTorchModelHubMixin):
    def __init__(self, args):
        super(OGNIDC, self).__init__()

        self.args = args
        self.GRU_iters = self.args.GRU_iters
        set_jacobi(not getattr(args, 'cg_no_precond', False))

        # The MA-depthmap prior always contributes a second depth channel
        # (mono prior + sparse depth) alongside RGB.
        depth_input_channels = 2

        self.backbone = Backbone(args, mode=self.args.backbone_mode, depth_input_channels=depth_input_channels)

        self.hdim = args.gru_hidden_dim
        self.cdim = args.gru_context_dim

        self.resolution = args.num_resolution

        # Repo-owned MA depthmap prior. Replaces the original DAv2 ViT-L path:
        # same I/O contract (ImageNet-normalized RGB in, disparity-polarity
        # depth out) so the rest of the forward pass is unchanged.
        self.depth_module = MADepthMapPrior(
            fp16=not getattr(args, 'prior_fp32', False),
            trt=getattr(args, 'trt', False),
            prior_batch_size=getattr(args, 'prior_batch_size', 1),
            reuse_first_in_batch=getattr(args, 'prior_reuse_first_in_batch', False),
            lazy_load=getattr(args, 'prior_lazy_load', False),
        ).eval()
        for param in self.depth_module.parameters():
            param.requires_grad = False

        # NLSPN
        self.prop_time = args.prop_time
        if args.spn_type == "nlspn":
            from nlspn_module import NLSPN

            self.num_neighbors = args.prop_kernel * args.prop_kernel - 1
            if self.prop_time > 0:
                self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                        self.args.prop_kernel)
        elif args.spn_type == "dyspn":
            from dyspn_module import DySPN_Module

            self.num_neighbors = 5
            if self.prop_time > 0:
                assert self.prop_time == 6
                self.prop_layer = DySPN_Module(iteration=self.prop_time,
                                               num=self.num_neighbors,
                                               mode='yx')
        else:
            raise NotImplementedError

        # DySPN

        self.downsample_rate = args.backbone_output_downsample_rate
        self.update_block = BasicUpdateBlock(args=self.args, resolution=self.resolution, hidden_dim=self.hdim,
                                             mask_r=self.downsample_rate,
                                             conf_min=self.args.conf_min)

    def initialize_depth(self, sparse_depth):
        log_depth_init = torch.zeros_like(sparse_depth)
        log_depth_grad_init = torch.zeros_like(sparse_depth).repeat(1, 2 * self.resolution, 1, 1)  # B x 2 x H x W

        return log_depth_init, log_depth_grad_init

    def forward(self, sample):
        rgb = sample['rgb']
        dep = torch.clone(sample['dep'])   # mutated below -> needs its own copy
        # dep_original is read-only here (median/scale, and DySPN uses it as a
        # fixed re-anchor target), so alias instead of a 2nd full copy (#3).
        # The end-to-end smoke test (full DySPN path, asserts anchor MAE)
        # guards against any accidental in-place write.
        dep_original = sample['dep']
        K = sample['K']
        depth_pattern = sample['pattern']

        B, _, H, W = rgb.shape

        # there are two sparse depths:
        # dep_integrator is scale-senstive, bringing the actual scale values to the depth integrator
        # dep_network_input is scale-agnostic, making the network invariant to depth scale changes
        # if you multiply the sparse depth by a factor s, the network is guaranteed to produce a
        # dense depth also multiplied by the factor s.

        valid_sparse_mask = (dep > 0.0).float()
        valid_sparse_mask_network_input = valid_sparse_mask

        # BasicUpdateBlock keeps K in its signature for compatibility but does
        # not read it, so avoid cloning/scaling it on the inference path.
        K_downsampled = K

        # this is full-res depth
        if self.args.whiten_sparse_depths:
            if getattr(self.args, 'capturable_inference', False):
                medians = _median_nonzero_flat(dep_original.reshape(B, -1))
            else:
                medians = torch.ones(B, device=rgb.device)
                for b in range(B):
                    nonzeros = dep_original[b] > 0.0
                    # #5: len(bool mask) is its dim-0 size (never 0); the original
                    # guard never fired and a zero-anchor sample would error in
                    # torch.median. Guard on the actual count instead.
                    if nonzeros.any():
                        medians[b] = torch.median(dep_original[b][nonzeros])

            dep_network_input = dep_original / medians.reshape(B, 1, 1, 1)  # make the median to be always 1.0
        else:
            dep_network_input = torch.clone(dep_original)

        # sparse depth needs downsample before feeding into the optim layer
        if self.downsample_rate > 1:
            if self.args.depth_downsample_method == "mean":
                dep = F.avg_pool2d(dep, self.downsample_rate)
                valid_sparse_mask = F.avg_pool2d(valid_sparse_mask, self.downsample_rate)
                dep[valid_sparse_mask > 0.0] = dep[valid_sparse_mask > 0.0] / valid_sparse_mask[valid_sparse_mask > 0.0]
                valid_sparse_mask[valid_sparse_mask > 0.0] = 1.0
            elif self.args.depth_downsample_method == "min":
                dep[dep == 0.0] = 100000.0  # set the invalid values to inf
                dep = -F.max_pool2d(-dep, self.downsample_rate)  # trick to do min-pooling
                valid_sparse_mask = F.max_pool2d(valid_sparse_mask,
                                                 self.downsample_rate)  # mask is 1 if at least one pt in neighbor
                dep[valid_sparse_mask == 0.0] = 0.0  # set invalid value back to 0.0, for safety
            else:
                raise NotImplementedError

        if self.args.depth_activation_format == "exp":
            dep_integrator = torch.log(dep)
            dep_network_input = torch.log(dep_network_input)
        else:
            dep_integrator = dep
            dep_network_input = dep_network_input

        dep_integrator[valid_sparse_mask == 0.0] = 0.0
        dep_network_input[valid_sparse_mask_network_input == 0.0] = 0.0

        if self.args.training_depth_random_shift_range > 0.0 and self.training:
            batch_size = rgb.shape[0]
            random_shift = torch.empty(batch_size).uniform_(-0.5,
                                                            0.5).cuda() * self.args.training_depth_random_shift_range
            dep_network_input = dep_network_input + random_shift.reshape(batch_size, 1, 1, 1)

        # MA-depthmap mono-depth prior (replaces the original DAv2 path).
        sky_mask = None  # (B,1,H,W) 1.0 = prior-flagged far-field/sky, else 0
        if 'mono_dep' in sample:
            mono_depth = sample['mono_dep']
        else:
            # #1: feed full-res RGB straight to the prior. MA-depthmap forces
            # its own 384x384 internally and resizes its output back to the
            # input size, so passing full-res does ONE pair of interpolations
            # (full->384->full) instead of the DAv2-vestige quadruple
            # (full->518->384->384->518->full). resize_image / 518 were ViT-14
            # DAv2 requirements the vendored DINOv3 prior doesn't share.

            # Per-sample anchor-derived cap: zero the prior wherever its metric
            # depth exceeds k x the farthest SfM anchor, right after MA
            # inference. The raw prior clamps open sky to ~1000 m; left in, that
            # skews the percentile-normalize below and feeds the backbone an
            # out-of-distribution channel. Caps from dep_original (full-res,
            # pre-downsample sparse depth).
            cap_factor = getattr(self.args, 'anchor_cap_factor', 0.0)
            max_metric_depth = None
            if cap_factor and cap_factor > 0:
                valid = dep_original > 0
                max_anchor = dep_original.masked_fill(~valid, 0.0).reshape(B, -1).amax(dim=1)
                max_metric_depth = torch.where(
                    valid.reshape(B, -1).any(dim=1),
                    cap_factor * max_anchor,
                    torch.full_like(max_anchor, float('inf')),
                )

            # disparity-polarity dense prediction (high = close), to match the
            # convention the backbone was trained against under DAv2. MA.infer
            # resizes its output back to the input size, so this is full-res.
            prior_disp = self.depth_module.forward(
                rgb, max_metric_depth=max_metric_depth
            )  # B x H x W — disparity, 0 exactly where capped as far/sky

            # Already full-res: exact 0s, no interpolation blur on the mask.
            if max_metric_depth is not None:
                sky_mask = (prior_disp == 0).float().unsqueeze(1)  # B x 1 x H x W

            depth_pred_raw = prior_disp.unsqueeze(1)  # B x 1 x H x W
            depth_pred_raw = F.relu(depth_pred_raw)

            # normalize to [0,1] via 2nd-98th percentile clip
            depth_flat = depth_pred_raw.reshape(B, -1)
            q_min, q_max = quantile_02_98_flat(depth_flat)
            _min = q_min.reshape(B, 1, 1, 1)
            _max = q_max.reshape(B, 1, 1, 1)
            mono_depth = (depth_pred_raw - _min) / (_max - _min).clamp_min(1e-6)

        dep_network_input = torch.cat([dep_network_input, mono_depth], dim=1)

        # A rare non-finite fp16 MA prior frame gets sanitized above. For that
        # captured shape, keep the decoder blocks in eager PyTorch: the B16 TRT
        # decoder kernels are much slower on the sanitized real-frame batch.
        forced_decoder_blocks = []
        if (
            ('mono_dep' in sample and rgb.shape[0] > 1)
            or getattr(self.depth_module, 'last_depth_had_nonfinite', False)
        ):
            for name in ('dec6', 'dec5', 'dec4', 'dec3', 'dec2'):
                block = getattr(self.backbone, name, None)
                if hasattr(block, 'force_eager'):
                    forced_decoder_blocks.append((block, block.force_eager))
                    block.force_eager = True

        # backbone
        assert self.args.pred_context_feature
        try:
            _, spn_guide, spn_confidence, context, confidence_input, confidence_output = self.backbone(
                rgb,
                dep_network_input,
                depth_pattern,
            )
        finally:
            for block, force_eager in forced_decoder_blocks:
                block.force_eager = force_eager

        if confidence_input is None:
            confidence_input = torch.ones_like(dep)  # B x 1 x H x W

        net, inp = torch.split(context, [self.hdim, self.cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # initialization
        log_depth_pred, log_depth_grad_pred_init = self.initialize_depth(dep)
        log_depth_grad_pred = log_depth_grad_pred_init

        # Dummy variable for recording gradients during training; inference
        # never consumes it, so avoid allocating a full-resolution zero tensor.
        b_init = torch.zeros_like(dep, requires_grad=True) if self.training else None

        log_depth_grad_predictions = []  # record the init value also
        confidence_predictions = []
        depth_predictions_up = []
        depth_predictions_up_initial = []

        resolution = self.resolution

        for itr in range(self.GRU_iters):
            log_depth_pred = log_depth_pred.detach()
            log_depth_grad_pred = log_depth_grad_pred.detach()

            # ideally, we should whiten the log_depth_pred, so that the input to gru is always invariant to depth scale.

            if itr == 0:
                log_depth_pred_whitened = log_depth_pred
            elif self.args.gru_internal_whiten_method == "mean":
                log_depth_pred_mean = torch.mean(log_depth_pred, dim=(1, 2, 3), keepdim=True)
                log_depth_pred_whitened = log_depth_pred - log_depth_pred_mean
            else:
                log_depth_pred_median = torch.median(log_depth_pred.reshape(B, -1), dim=1)[0]
                log_depth_pred_whitened = log_depth_pred - log_depth_pred_median.reshape(B, 1, 1, 1)

            net, up_mask, delta_log_depth_grad, weights_depth_grad, weights_input = self.update_block(net, inp,
                                                                                                      log_depth_pred_whitened,
                                                                                                      log_depth_grad_pred,
                                                                                                      K_downsampled
                                                                                                      )
            # print('depth grad pred', log_depth_grad_pred)
            log_depth_grad_pred = log_depth_grad_pred + delta_log_depth_grad

            # numerical stability
            thres = self.args.optim_layer_input_clamp
            log_depth_grad_pred = torch.clamp(log_depth_grad_pred, min=-thres, max=thres)

            # the optimization layer use the prediction from last round to accelerate convergence
            # The optim layer currently uses only sparse input-confidence
            # channel 0, so avoid broadcasting it to all resolution channels.
            if self.args.multi_resolution_learnable_input_weights:
                input_confidence = confidence_input * weights_input[:, :1]
            else:
                input_confidence = confidence_input

            log_depth_pred, b_init = DepthGradOptimLayer.apply(log_depth_grad_pred,
                                                               dep_integrator,
                                                               valid_sparse_mask,
                                                               weights_depth_grad,
                                                               input_confidence,
                                                               resolution,
                                                               log_depth_pred,
                                                               b_init,
                                                                self.args.integration_alpha,
                                                                getattr(self.args, 'cg_rtol', 1e-5),
                                                                (getattr(self.args, 'cg_fixed_iters', 0)
                                                                 or getattr(self.args, 'cg_maxiter', 5000)),
                                                                getattr(self.args, 'cg_check_interval', 1),
                                                                getattr(self.args, 'cg_fixed_iters', 0) > 0)

            log_depth_grad_predictions.append(log_depth_grad_pred)
            confidence_predictions.append(weights_depth_grad)

            # convex upsample
            if self.downsample_rate > 1:
                log_depth_up = upsample_depth(log_depth_pred, up_mask, r=self.downsample_rate)
            else:
                log_depth_up = log_depth_pred

            # in case where Hrgb / downsample_rate is not integer, extra interpolation is needed
            _, _, Hrgb, Wrgb = rgb.shape
            _, _, Hd, Wd = log_depth_up.shape
            if Hd != Hrgb or Wd != Wrgb:
                print('warning: dim mismatch!')
                log_depth_up = F.interpolate(log_depth_up, size=(Hrgb, Wrgb), mode='bilinear', align_corners=True)

            if self.args.depth_activation_format == "exp":
                depth_pred_up_init = torch.exp(log_depth_up)
            else:
                depth_pred_up_init = log_depth_up

            depth_predictions_up_initial.append(depth_pred_up_init)

            # SPN
            if self.prop_time > 0 and (self.training or itr == self.GRU_iters - 1):
                if self.args.spn_type == "dyspn":
                    spn_out = self.prop_layer(depth_pred_up_init,
                                              spn_guide,
                                              dep_original,
                                              spn_confidence)
                    depth_pred_up_final = spn_out['pred']
                    dyspn_offset = spn_out['offset']
                elif self.args.spn_type == "nlspn":
                    depth_pred_up_final, _, _, _, _ = self.prop_layer(depth_pred_up_init, spn_guide, spn_confidence,
                                                                      None)
                    dyspn_offset = None
            else:
                depth_pred_up_final = depth_pred_up_init
                dyspn_offset = None

            depth_predictions_up.append(depth_pred_up_final)

        output = {'pred': depth_predictions_up[-1], 'pred_inter': depth_predictions_up,
                  'depth_predictions_up_initial': depth_predictions_up_initial,
                  'log_depth_grad_inter': log_depth_grad_predictions,
                  'log_depth_grad_init': log_depth_grad_pred_init,
                  'confidence_depth_grad_inter': confidence_predictions,
                  'dep_down': dep,
                  'confidence_input': confidence_input,
                  'confidence_output': confidence_output,
                  'dyspn_offset': dyspn_offset,
                  'sky_mask': sky_mask,
                  }

        return output
