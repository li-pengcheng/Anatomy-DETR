"""
Anatomy-DETR decoder components, built upon DEIM / D-FINE.
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

from .dfine_utils import weighting_function, distance2bbox
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from .tooth_prior import ToothPriorBuilder, EncoderPriorAttentionAdapter, gt_hard_weights_for_target
from ..core import register

__all__ = ['DFINETransformer']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




def _inv_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-4)
    return torch.log(torch.expm1(value))


def _logit_prob(p: float) -> float:
    p = max(1e-4, min(1.0 - 1e-4, float(p)))
    return math.log(p / (1.0 - p))


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method='default',
        offset_scale=0.5,
        prior_bias_num_heads=None,
        prior_bias_gate_init=0.7,
        prior_bias_gate_learnable=False,
        prior_bias_clip=0.7,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        # [stat_risk, hard_miss, image_risk, tooth_inside, side_context] soft-bias gates.
        self.prior_bias_raw = nn.Parameter(_inv_softplus(torch.tensor([0.5, 0.5, 1.0, 0.5, 1.0], dtype=torch.float32)))

        # Query groups: 0=global, 1=risk, 2=tooth, 3=context (DN queries use id -1 -> no effect).
        self.num_query_groups = 4
        # #7 per-group, zero-init learnable step that nudges sampling points along the risk gradient.
        # Zero init => no displacement at the start, so behaviour matches the baseline exactly.
        self.sampling_pos_step = nn.Parameter(torch.zeros(self.num_query_groups, dtype=torch.float32))
        self.sampling_pos_eps = 0.01
        self.sampling_pos_max_move = 0.05
        # #10 per-group, log-space learnable multiplier on offset_scale (deformable sampling radius).
        # exp(0)=1 keeps stat/risk/tooth at the base radius; context starts ~1.5x wider so it can
        # reach across the inter-proximal contact to the neighbouring tooth (proximal caries).
        offset_scale_logit = torch.zeros(self.num_query_groups, dtype=torch.float32)
        offset_scale_logit[3] = math.log(1.5)
        self.offset_scale_logit = nn.Parameter(offset_scale_logit)

        # Decoder prior head split.  Only the heads marked by prior_head_mask receive
        # tooth/risk/context attention bias; all other heads stay image-only.  We select the prior
        # heads uniformly across the head index space (e.g. 8 heads, k=2 -> heads 0 and 4) instead of
        # taking consecutive heads, so prior-guided and image-only heads are interleaved.
        if prior_bias_num_heads is None:
            prior_bias_num_heads = num_heads
        prior_bias_num_heads = int(max(0, min(num_heads, prior_bias_num_heads)))
        head_mask = torch.zeros(num_heads, dtype=torch.float32)
        if prior_bias_num_heads >= num_heads:
            head_mask.fill_(1.0)
        elif prior_bias_num_heads > 0:
            idx = torch.floor(
                torch.arange(prior_bias_num_heads, dtype=torch.float32) * float(num_heads) / float(prior_bias_num_heads)
            ).long().clamp(max=num_heads - 1)
            head_mask[idx] = 1.0
        self.register_buffer('prior_head_mask', head_mask, persistent=False)

        self.prior_bias_clip = float(prior_bias_clip)
        self.head_prior_gate_raw = nn.Parameter(
            torch.full((num_heads,), _logit_prob(prior_bias_gate_init), dtype=torch.float32),
            requires_grad=bool(prior_bias_gate_learnable)
        )

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int],
                prior_maps: torch.Tensor = None,
                query_groups: torch.Tensor = None,
                prior_bias_scale: float = 1.0,
                enable_sampling_pos_bias: bool = True):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_logits = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            # #10 per-group learnable sampling radius (context wider); falls back to base scale.
            group_scale = self._offset_scale_per_query(query_groups, bs, Len_q, query.dtype, query.device)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale * group_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        # #7 nudge sampling positions of risk/tooth/context queries up the risk gradient.
        # #7a only the first few decoder layers run this (controlled by the caller); deeper layers
        # have near-converged reference points where the finite-difference cost is not worth it.
        if enable_sampling_pos_bias:
            sampling_locations = self._apply_sampling_pos_bias(
                sampling_locations, prior_maps, query_groups, value_spatial_shapes)
        else:
            # Keep this layer's sampling_pos_step in the autograd graph (zero contribution) so DDP
            # with find_unused_parameters=False does not flag it as unused where the nudge is off.
            sampling_locations = sampling_locations + 0.0 * self.sampling_pos_step.sum()

        prior_bias = self._build_prior_bias(prior_maps, query_groups, sampling_locations, value_spatial_shapes,
                                            prior_bias_scale=prior_bias_scale)
        if prior_bias is not None:
            if self.prior_bias_clip > 0:
                prior_bias = prior_bias.clamp(max=self.prior_bias_clip)
            head_gate = torch.sigmoid(self.head_prior_gate_raw).to(attention_logits.dtype)
            head_mask = self.prior_head_mask.to(device=attention_logits.device, dtype=attention_logits.dtype)
            effective_head_gate = head_gate * head_mask
            attention_logits = attention_logits + effective_head_gate.view(1, 1, -1, 1) * prior_bias.to(attention_logits.dtype)
        attention_weights = F.softmax(attention_logits, dim=-1)

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        return output
    

    def _sample_prior_maps(self, prior_maps: torch.Tensor, sampling_locations: torch.Tensor, value_spatial_shapes):
        if prior_maps is None or sampling_locations is None:
            return None
        bs, len_q, n_heads, _, _ = sampling_locations.shape
        prior_maps = prior_maps.detach().to(
            device=sampling_locations.device,
            dtype=torch.float32
        )
        channels = prior_maps.shape[-1]
        level_sizes = [int(h) * int(w) for h, w in value_spatial_shapes]
        prior_levels = prior_maps.split(level_sizes, dim=1)
        loc_levels = sampling_locations.split(self.num_points_list, dim=3)
        sampled_levels = []
        for lvl, (shape, loc_l, prior_l) in enumerate(zip(value_spatial_shapes, loc_levels, prior_levels)):
            h, w = int(shape[0]), int(shape[1])
            n_pts = int(loc_l.shape[3])
            with torch.autocast(device_type='cuda', enabled=False):
                prior_img = prior_l.permute(0, 2, 1).reshape(bs, channels, h, w).float()
                if self.method == 'default':
                    grid = 2.0 * loc_l.float() - 1.0
                else:
                    grid = loc_l.float()
                grid = grid.reshape(bs, len_q * n_heads * n_pts, 1, 2)
                sampled = F.grid_sample(
                    prior_img,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )
                sampled = sampled.squeeze(-1).permute(0, 2, 1).reshape(
                    bs, len_q, n_heads, n_pts, channels
                )
            sampled_levels.append(sampled)
        return torch.cat(sampled_levels, dim=3)


    def _build_prior_bias(self, prior_maps: torch.Tensor, query_groups: torch.Tensor, sampling_locations: torch.Tensor, value_spatial_shapes, prior_bias_scale: float = 1.0):
        if prior_maps is None or query_groups is None:
            return None
        sampled = self._sample_prior_maps(prior_maps, sampling_locations, value_spatial_shapes)
        if sampled is None or sampled.shape[-1] < 5:
            return None
        # #5 epoch warmup: gates ramp in from 0 so early (random) encoder scores are not over-steered.
        gates = F.softplus(self.prior_bias_raw).to(device=sampled.device, dtype=sampled.dtype)
        scale = float(prior_bias_scale)
        if scale != 1.0:
            gates = gates * scale
        stat, hard, img, tooth, context = [sampled[..., i] for i in range(5)]
        groups = query_groups.to(device=sampled.device).long().view(sampled.shape[0], sampled.shape[1], 1, 1)
        bias = torch.zeros_like(stat)
        risk_mask = groups == 1
        tooth_mask = groups == 2
        context_mask = groups == 3
        bias = torch.where(risk_mask, gates[0] * stat + gates[1] * hard + gates[2] * img + gates[3] * tooth, bias)
        bias = torch.where(tooth_mask, gates[3] * tooth, bias)
        bias = torch.where(context_mask, gates[4] * context, bias)
        return bias

    def _offset_scale_per_query(self, query_groups, bs, Len_q, dtype, device):
        """#10 [bs, Len_q, 1, 1, 1] per-group multiplier exp(logit[group]); 1.0 for global/DN/missing."""
        if query_groups is None:
            return torch.ones((1, 1, 1, 1, 1), dtype=dtype, device=device)
        factors = torch.exp(self.offset_scale_logit).to(device=device, dtype=dtype)
        g = query_groups.to(device=device).long()
        valid = g >= 0  # DN queries (-1) keep the base scale
        idx = g.clamp(min=0, max=self.num_query_groups - 1)
        scale = factors[idx]
        scale = torch.where(valid, scale, torch.ones_like(scale))
        return scale.view(bs, Len_q, 1, 1, 1)

    def _apply_sampling_pos_bias(self, sampling_locations, prior_maps, query_groups, value_spatial_shapes):
        """#7 move sampling points of risk/tooth/context queries along the (detached) risk gradient.

        The per-group step is zero-initialized, so this is a strict no-op at the start of training and
        whenever no prior maps / query groups are available.  Global (0) and DN (-1) queries never move.
        """
        if prior_maps is None or query_groups is None or prior_maps.shape[-1] < 1:
            return sampling_locations
        bs, Len_q = sampling_locations.shape[:2]
        g = query_groups.to(device=sampling_locations.device).long()
        step = self.sampling_pos_step.to(device=sampling_locations.device, dtype=sampling_locations.dtype)
        idx = g.clamp(min=0, max=self.num_query_groups - 1)
        per_q = step[idx]
        per_q = torch.where(g > 0, per_q, torch.zeros_like(per_q))  # only risk/tooth/context move
        # NOTE: do not early-return on a zero step value -- keeping the parameter in the graph every
        # iteration (move == 0 at init) guarantees it receives gradient under DDP find_unused=False.
        eps = float(self.sampling_pos_eps)
        risk = prior_maps[..., 0:1]  # risk_prior_norm channel, detached inside _sample_prior_maps

        def sample(locs):
            return self._sample_prior_maps(risk, locs.clamp(0.0, 1.0), value_spatial_shapes)[..., 0]

        ex = torch.zeros_like(sampling_locations); ex[..., 0] = eps
        ey = torch.zeros_like(sampling_locations); ey[..., 1] = eps
        gx = (sample(sampling_locations + ex) - sample(sampling_locations - ex)) / (2.0 * eps)
        gy = (sample(sampling_locations + ey) - sample(sampling_locations - ey)) / (2.0 * eps)
        grad = torch.stack([gx, gy], dim=-1)  # [bs, Len_q, n_heads, n_points, 2]
        move = per_q.view(bs, Len_q, 1, 1, 1) * grad
        # Keep image-only heads truly image-only: they neither receive prior attention bias nor
        # prior-gradient sampling-position nudges.
        head_mask = self.prior_head_mask.to(device=sampling_locations.device, dtype=sampling_locations.dtype)
        move = move * head_mask.view(1, 1, self.num_heads, 1, 1)
        move = move.clamp(-float(self.sampling_pos_max_move), float(self.sampling_pos_max_move))
        return (sampling_locations + move).clamp(0.0, 1.0)


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None,
                 prior_bias_num_heads=None,
                 prior_bias_gate_init=0.7,
                 prior_bias_gate_learnable=False,
                 prior_bias_clip=0.7):
        super(TransformerDecoderLayer, self).__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points,
            method=cross_attn_method,
            prior_bias_num_heads=prior_bias_num_heads,
            prior_bias_gate_init=prior_bias_gate_init,
            prior_bias_gate_learnable=prior_bias_gate_learnable,
            prior_bias_clip=prior_bias_clip,
        )
        self.dropout2 = nn.Dropout(dropout)

        # gate
        self.gateway = Gate(d_model)

        # Optimization 2: proximal-contrast readout for risk/context queries.  Proximal caries sit
        # across the inter-proximal contact, so the discriminative cue is the *difference* between
        # the two local-u sides of a query.  We sample the highest-resolution value map at center
        # +/- a learnable offset along the tooth-local u axis estimated from the mask-PCA rbox, then
        # feed the difference through a zero-init projection.  It remains a strict no-op until
        # trained and only affects risk (group 1) / context (group 3) queries.
        self.n_head = n_head
        self.d_model = d_model
        self.contrast_delta = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))
        self.contrast_proj = nn.Linear(d_model, d_model)
        init.constant_(self.contrast_proj.weight, 0.0)
        init.constant_(self.contrast_proj.bias, 0.0)

        # #4 contralateral negative reference.  Caries is usually asymmetric: the same-type tooth on
        # the OTHER side of the anatomical midline is, more often than not, healthy and therefore a
        # strong negative reference.  For risk/tooth/context queries we sample the highest-resolution
        # value map at the query center and at its mirror across the tracked midline, and feed the
        # difference (self - contralateral) through a zero-init projection.  Strict no-op until trained.
        self.contra_proj = nn.Linear(d_model, d_model)
        init.constant_(self.contra_proj.weight, 0.0)
        init.constant_(self.contra_proj.bias, 0.0)

        # #5 intra-tooth self-attention branch.  On top of the normal global self-attention, queries
        # that belong to the SAME tooth exchange information through a dedicated MHA restricted to the
        # same-tooth block.  Gated by a zero-init scalar so the branch is a strict no-op at the start
        # of training and the baseline is reproduced exactly until it learns to help (e.g. per-tooth
        # de-duplication and consistent single-lesion decisions).
        self.intra_tooth_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.intra_tooth_norm = nn.LayerNorm(d_model)
        self.intra_tooth_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def _contrast_readout(self, value, reference_points, query_groups, spatial_shapes, query_u_axes=None):
        """Difference of the two local-u neighbours of each risk/context query.

        For a selected query, ``query_u_axes`` stores the rbox-derived mesial/distal unit vector in
        normalized image coordinates.  Sampling along this local axis is more faithful than a fixed
        horizontal offset on tilted or curved panoramic teeth.  If the axis is unavailable, we fall
        back to the old horizontal direction for compatibility.
        """
        if query_groups is None:
            return None
        g = query_groups.long()
        mask = (g == 1) | (g == 3)
        if not bool(mask.any()):
            return None
        v0 = value[0]  # [bs, n_head, head_dim, H0*W0]
        bs = v0.shape[0]
        if int(v0.shape[1]) * int(v0.shape[2]) != int(self.d_model):
            return None
        h0, w0 = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
        feat = v0.reshape(bs, self.d_model, h0, w0).float()
        center = reference_points[:, :, 0, :2].float()  # [bs, lq, 2] in [0,1]

        if query_u_axes is not None and query_u_axes.shape[:2] == center.shape[:2]:
            axes = query_u_axes.to(device=center.device, dtype=center.dtype).float()
            norm = axes.norm(dim=-1, keepdim=True)
            fallback = torch.zeros_like(axes)
            fallback[..., 0] = 1.0
            axes = torch.where(norm > 1e-6, axes / norm.clamp(min=1e-6), fallback)
            off = axes * self.contrast_delta.to(center.dtype)
        else:
            off = torch.zeros_like(center)
            off[..., 0] = self.contrast_delta.to(center.dtype)

        def _sample(points):
            grid = (points.clamp(0.0, 1.0) * 2.0 - 1.0).unsqueeze(2)  # [bs, lq, 1, 2]
            s = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=False)
            return s.squeeze(-1).permute(0, 2, 1)  # [bs, lq, d_model]

        contrast = (_sample(center + off) - _sample(center - off))
        contrast = contrast * mask.to(contrast.dtype).unsqueeze(-1)
        return self.contrast_proj(contrast.to(self.contrast_proj.weight.dtype))

    def _contralateral_readout(self, value, reference_points, query_groups, spatial_shapes, query_midline_x=None):
        """#4 difference between a query's own features and its contralateral mirror across the midline.

        For risk/tooth/context queries, the mirror of normalized center (x, y) across the anatomical
        midline ``m`` is (2m - x, y).  We sample the highest-resolution value map at both locations and
        return proj(self - contralateral).  Mirrors land on the same-type tooth on the other arch side,
        which is usually healthy, so the difference is a strong asymmetry cue for caries.  Zero-init
        projection -> strict no-op until trained; only risk(1)/tooth(2)/context(3) queries contribute.
        """
        if query_groups is None or query_midline_x is None:
            return None
        g = query_groups.long()
        mask = (g == 1) | (g == 2) | (g == 3)
        # NOTE: we deliberately do NOT early-return when ``mask`` is all-False.  Running the readout
        # unconditionally (and zeroing the contribution via ``mask`` below) keeps contra_proj in the
        # autograd graph every step so DDP(find_unused_parameters=False) is safe on all-global batches.
        v0 = value[0]
        bs = v0.shape[0]
        if int(v0.shape[1]) * int(v0.shape[2]) != int(self.d_model):
            return None
        h0, w0 = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
        feat = v0.reshape(bs, self.d_model, h0, w0).float()
        center = reference_points[:, :, 0, :2].float()                     # [bs, lq, 2] in [0,1]
        m = query_midline_x.to(device=center.device, dtype=center.dtype)   # [bs, lq]
        mirror = center.clone()
        mirror[..., 0] = (2.0 * m - center[..., 0])                        # reflect x across midline

        def _sample(points):
            grid = (points.clamp(0.0, 1.0) * 2.0 - 1.0).unsqueeze(2)
            s = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=False)
            return s.squeeze(-1).permute(0, 2, 1)

        diff = (_sample(center) - _sample(mirror))
        diff = diff * mask.to(diff.dtype).unsqueeze(-1)
        return self.contra_proj(diff.to(self.contra_proj.weight.dtype))

    def forward(self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None,
                prior_maps=None,
                query_groups=None,
                prior_bias_scale=1.0,
                group_pos=None,
                uv_pos=None,
                query_u_axes=None,
                enable_sampling_pos_bias=True,
                query_midline_x=None,
                intra_tooth_mask=None):

        # #8 group positional embedding (self-attn only) lets each query know its role so the four
        # query types stop competing for the same positions.  #9 the tooth-local (u,v) prior is added
        # to both self- and cross-attn positions.  Both are zero at init and skipped on a dim mismatch
        # (wider refinement layers), so the baseline is reproduced exactly until they learn.
        self_pos = query_pos_embed
        cross_pos = query_pos_embed
        if uv_pos is not None and query_pos_embed is not None and uv_pos.shape[-1] == query_pos_embed.shape[-1]:
            uv_pos = uv_pos.to(query_pos_embed.dtype)
            self_pos = self_pos + uv_pos
            cross_pos = cross_pos + uv_pos
        if group_pos is not None and query_pos_embed is not None and group_pos.shape[-1] == query_pos_embed.shape[-1]:
            self_pos = self_pos + group_pos.to(query_pos_embed.dtype)

        # self attention
        q = k = self.with_pos_embed(target, self_pos)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # #5 intra-tooth self-attention branch (gated, zero-init no-op at start).  Queries of the same
        # tooth refine each other in a dedicated attention restricted to the same-tooth block.  The
        # branch ALWAYS runs so its parameters always receive gradient under DDP
        # (find_unused_parameters=False): with a same-tooth mask it restricts attention to each tooth
        # block; without one (a batch that happens to carry no tooth ids) it falls back to ordinary
        # full attention.  Either way the zero-init gate fully bypasses the branch (including its norm),
        # so the baseline activations are reproduced exactly until the gate learns to open.
        q2 = self.with_pos_embed(target, self_pos)
        intra_attn_mask = None
        if intra_tooth_mask is not None:
            # bool attn_mask convention for nn.MultiheadAttention: True == NOT allowed to attend.
            # The diagonal stays allowed (intra_tooth_mask keeps the eye), so no row is fully masked
            # and the result is NaN-free.  Expand [bs, Lq, Lq] -> [bs*n_head, Lq, Lq].
            bool_mask = ~intra_tooth_mask.to(device=target.device)
            bs, lq, _ = bool_mask.shape
            intra_attn_mask = bool_mask.unsqueeze(1).expand(bs, self.n_head, lq, lq).reshape(bs * self.n_head, lq, lq)
        intra2, _ = self.intra_tooth_attn(q2, q2, value=target, attn_mask=intra_attn_mask)
        target = target + self.intra_tooth_gate.to(target.dtype) * self.intra_tooth_norm(intra2)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, cross_pos),
            reference_points,
            value,
            spatial_shapes,
            prior_maps=prior_maps,
            query_groups=query_groups,
            prior_bias_scale=prior_bias_scale,
            enable_sampling_pos_bias=enable_sampling_pos_bias)

        target = self.gateway(target, self.dropout2(target2))

        # Optimization 2: proximal-contrast readout (risk/context queries only; zero-init no-op start).
        contrast = self._contrast_readout(value, reference_points, query_groups, spatial_shapes, query_u_axes=query_u_axes)
        if contrast is not None:
            target = target + contrast.to(target.dtype)

        # #4 contralateral negative-reference readout (risk/tooth/context; zero-init no-op start).
        contra = self._contralateral_readout(value, reference_points, query_groups, spatial_shapes,
                                             query_midline_x=query_midline_x)
        if contra is not None:
            target = target + contra.to(target.dtype)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act='relu'):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max+1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act='relu', num_tooth_types=6, num_classes=1,
                 sampling_pos_max_layers=3, decoder_prior_bias_max_layers=None):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        # #7a only the first ``sampling_pos_max_layers`` layers run the finite-difference sampling-
        # position nudge (the expensive part); deeper layers have near-converged references.
        self.sampling_pos_max_layers = int(sampling_pos_max_layers)
        # Only the first N decoder layers are allowed to use prior attention bias.  Later layers
        # become image-driven refinement layers for classification/ranking and precise localization.
        self.decoder_prior_bias_max_layers = (
            self.num_layers if decoder_prior_bias_max_layers is None else int(decoder_prior_bias_max_layers)
        )
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] \
                    + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])

        # #8 group positional embedding: 0=global, 1=risk, 2=tooth, 3=context, and a dedicated row for
        # denoising queries (id -1 -> num_query_groups).  Zero-init -> no effect until trained.
        self.num_query_groups = 4
        self.group_pos_embed = nn.Embedding(self.num_query_groups + 1, hidden_dim)
        init.constant_(self.group_pos_embed.weight, 0.0)
        # #9 tooth-local (u, v) position prior: Fourier features -> MLP, zero-init last layer so it
        # starts as a no-op.  Injected only for risk/tooth/context queries (global/DN are masked out).
        self.uv_pos_num_freqs = 6
        self.uv_pos_proj = MLP(2 * 2 * self.uv_pos_num_freqs, hidden_dim, hidden_dim, 2, act=act)
        init.constant_(self.uv_pos_proj.layers[-1].weight, 0.0)
        init.constant_(self.uv_pos_proj.layers[-1].bias, 0.0)
        self.register_buffer('uv_pos_freqs',
                             (2.0 ** torch.arange(self.uv_pos_num_freqs, dtype=torch.float32)) * math.pi,
                             persistent=False)

        # NOTE (design change): the previous "tooth-conditioned classification bias" MLP was REMOVED.
        # It added a prior-driven additive term directly to the classification logits, which let the
        # tooth-type/(u,v) prior inflate or suppress caries confidence WITHOUT any supporting image
        # evidence.  That conflicts with the core principle that the decoder must decide caries from
        # image features, using the prior only to (a) steer which positions enter the decoder and
        # (b) guide where cross-attention samples.  All prior influence on the *score* is therefore
        # confined to query selection (encoder stage) and to attention sampling/aggregation; the
        # final per-query logit now comes purely from the query's image-conditioned content.
        self.num_tooth_types = int(num_tooth_types)
        self.cls_bias_num_classes = int(num_classes)

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def _group_pos(self, query_groups):
        """#8 [bs, Len_q, hidden] group positional embedding; DN (-1) maps to the dedicated last row."""
        if query_groups is None:
            return None
        g = query_groups.long()
        idx = torch.where(g < 0, torch.full_like(g, self.num_query_groups), g)
        idx = idx.clamp(min=0, max=self.num_query_groups)
        return self.group_pos_embed(idx)

    def _uv_pos(self, query_uv, query_groups):
        """#9 [bs, Len_q, hidden] tooth-local (u,v) prior, masked to risk/tooth/context queries only.

        The MLP runs on every query (so its weights always receive gradient under DDP); the result is
        masked to zero for global/DN queries, which carry no meaningful tooth-local coordinate.
        """
        if query_uv is None:
            return None
        uv = query_uv.to(dtype=self.group_pos_embed.weight.dtype).clamp(0.0, 1.0)
        freqs = self.uv_pos_freqs.to(device=uv.device, dtype=uv.dtype)
        ang = uv.unsqueeze(-1) * freqs.view(1, 1, 1, -1)            # [bs, Len_q, 2, F]
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(2)  # [bs, Len_q, 2*2*F]
        uv_pos = self.uv_pos_proj(feat)
        if query_groups is not None:
            mask = (query_groups.long() > 0).to(dtype=uv_pos.dtype).unsqueeze(-1)  # exclude global(0)/DN(-1)
            uv_pos = uv_pos * mask
        return uv_pos

    def _intra_tooth_mask(self, query_tooth):
        """#5 [bs, Len_q, Len_q] boolean mask, True where two queries belong to the SAME tooth.

        Queries with tooth id < 0 (global / DN / context-without-tooth) form no block.  The diagonal
        is kept True so a query always attends to itself in the intra-tooth branch.  Returns None when
        no tooth ids are available so the branch is skipped entirely.
        """
        if query_tooth is None:
            return None
        t = query_tooth.long()                                  # [bs, Len_q]
        same = (t.unsqueeze(-1) == t.unsqueeze(-2))             # [bs, Len_q, Len_q]
        valid = (t >= 0).unsqueeze(-1) & (t >= 0).unsqueeze(-2)
        same = same & valid
        # Keep the diagonal so every query can attend to itself; this also guarantees no fully-masked
        # row (which would NaN) and keeps the intra-tooth params in the autograd graph every step, so
        # DDP with find_unused_parameters=False never trips even on a (rare) all-global batch.
        eye = torch.eye(t.shape[1], device=t.device, dtype=torch.bool).unsqueeze(0)
        return same | eye

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                target,
                ref_points_unact,
                memory,
                spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                pre_bbox_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                dn_meta=None,
                prior_maps=None,
                query_groups=None,
                prior_bias_scale=1.0,
                query_uv=None,
                query_u_axes=None,
                query_type=None,
                query_midline_x=None,
                query_tooth=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        # #8/#9 precompute the role and tooth-local positional embeddings once (constant across layers).
        group_pos = self._group_pos(query_groups)
        uv_pos = self._uv_pos(query_uv, query_groups)
        # #5 intra-tooth self-attention mask (constant across layers).  This is an ADDITIVE block-
        # diagonal allow-mask combined with the existing (denoising) attn_mask so that, on top of the
        # normal global self-attention, queries belonging to the SAME tooth can exchange information
        # without being diluted by the hundreds of other queries.  It is realized through a separate
        # gated attention branch in the layer; here we only precompute the boolean same-tooth mask.
        intra_tooth_mask = self._intra_tooth_mask(query_tooth)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # TODO Adjust scale if needed for detachable wider layers
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            use_decoder_prior = i < self.decoder_prior_bias_max_layers
            layer_prior_bias_scale = prior_bias_scale if use_decoder_prior else 0.0
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed,
                           prior_maps=prior_maps, query_groups=query_groups, prior_bias_scale=layer_prior_bias_scale,
                           group_pos=group_pos, uv_pos=uv_pos, query_u_axes=query_u_axes,
                           enable_sampling_pos_bias=(use_decoder_prior and i < self.sampling_pos_max_layers),
                           query_midline_x=query_midline_x, intra_tooth_mask=intra_tooth_mask)

            if i == 0 :
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
               torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores


@register()
class DFINETransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 risk_query_cfg=None,
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 mlp_act='relu',
                 ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale*hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        assert query_select_method in ('default', 'one2many', 'agnostic', 'risk_group'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        self.risk_query_cfg = risk_query_cfg or {}
        self.risk_query_enabled = bool(self.risk_query_cfg.get('enabled', query_select_method == 'risk_group'))
        self.num_global_queries = int(self.risk_query_cfg.get('num_global_queries', 100))
        self.num_risk_queries = int(self.risk_query_cfg.get('num_risk_queries', 150))
        self.num_tooth_queries = int(self.risk_query_cfg.get('num_tooth_queries', 200))
        self.num_context_queries = int(self.risk_query_cfg.get('num_context_queries', 50))
        self.min_tooth_queries_per_tooth = int(self.risk_query_cfg.get('min_tooth_queries_per_tooth', 2))
        self.tooth_extra_eps = float(self.risk_query_cfg.get('tooth_extra_eps', 0.05))
        self.tooth_extra_gamma = float(self.risk_query_cfg.get('tooth_extra_gamma', 1.0))
        self.caries_class_index = int(self.risk_query_cfg.get('caries_class_index', 0))
        self.use_score_only_encoder_adapter = bool(self.risk_query_cfg.get('use_score_only_encoder_adapter', True))
        # #5 prior-bias gate warmup: gates ramp from 0 to full over this many epochs (0/None => no warmup).
        self.prior_bias_warmup_epochs = float(self.risk_query_cfg.get('prior_bias_warmup_epochs', 0.0) or 0.0)
        # Decoder prior is no longer applied to all heads/layers by default.  These knobs keep most
        # heads image-only and restrict prior guidance to the early search/refinement layers.
        self.decoder_prior_bias_strength = float(self.risk_query_cfg.get('decoder_prior_bias_strength', 1.0))
        self.decoder_prior_bias_num_heads = int(self.risk_query_cfg.get('decoder_prior_bias_num_heads', nhead))
        self.decoder_prior_bias_gate_init = float(self.risk_query_cfg.get('decoder_prior_bias_gate_init', 0.7))
        self.decoder_prior_bias_gate_learnable = bool(self.risk_query_cfg.get('decoder_prior_bias_gate_learnable', False))
        self.decoder_prior_bias_clip = float(self.risk_query_cfg.get('decoder_prior_bias_clip', 0.7))
        self.decoder_prior_bias_max_layers = int(self.risk_query_cfg.get('decoder_prior_bias_max_layers', num_layers))
        # #6 hard-miss hyper-parameters (surfaced for tuning); confidence_k / ema_momentum feed the risk map,
        # loss_beta drives the positive-loss modulation, update_interval is consumed by the solver.
        self.hard_miss_confidence_k = float(self.risk_query_cfg.get('hard_miss_confidence_k', 10.0))
        self.hard_miss_ema_momentum = float(self.risk_query_cfg.get('hard_miss_ema_momentum', 0.9))
        self.hard_miss_loss_beta = float(self.risk_query_cfg.get('hard_miss_loss_beta', 0.0))
        # #7a how many leading decoder layers run the (expensive) sampling-position nudge.
        self.sampling_pos_max_layers = int(self.risk_query_cfg.get('sampling_pos_max_layers', 3))
        self.num_tooth_types_cfg = int(self.risk_query_cfg.get('num_tooth_types', 6))
        self.register_buffer('current_epoch', torch.zeros((), dtype=torch.float32), persistent=True)
        if query_select_method == 'risk_group':
            assert self.num_global_queries + self.num_risk_queries + self.num_tooth_queries + self.num_context_queries == num_queries, \
                'risk_group query counts must sum to num_queries'

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout,
            activation, num_levels, num_points, cross_attn_method=cross_attn_method,
            prior_bias_num_heads=self.decoder_prior_bias_num_heads,
            prior_bias_gate_init=self.decoder_prior_bias_gate_init,
            prior_bias_gate_learnable=self.decoder_prior_bias_gate_learnable,
            prior_bias_clip=self.decoder_prior_bias_clip,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout,
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale,
            prior_bias_num_heads=self.decoder_prior_bias_num_heads,
            prior_bias_gate_init=self.decoder_prior_bias_gate_init,
            prior_bias_gate_learnable=self.decoder_prior_bias_gate_learnable,
            prior_bias_clip=self.decoder_prior_bias_clip,
        )
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
                                          reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation,
                                          num_tooth_types=self.num_tooth_types_cfg, num_classes=num_classes,
                                          sampling_pos_max_layers=self.sampling_pos_max_layers,
                                          decoder_prior_bias_max_layers=self.decoder_prior_bias_max_layers)
      # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)

        if self.risk_query_enabled:
            self.tooth_prior_builder = ToothPriorBuilder(
                template_size=int(self.risk_query_cfg.get('risk_template_size', 16)),
                context_expand_ratio=float(self.risk_query_cfg.get('tooth_expanded_box_ratio', self.risk_query_cfg.get('context_expand_ratio', 1.1))),
                context_side_expand=float(self.risk_query_cfg.get('context_side_expand', 0.05)),
                context_min_cell=int(self.risk_query_cfg.get('context_min_cell', 1)),
                num_tooth_types=int(self.risk_query_cfg.get('num_tooth_types', 6)),
                use_tooth_rotated_box=bool(self.risk_query_cfg.get('use_tooth_rotated_box', True)),
                tooth_rbox_quantile_low=float(self.risk_query_cfg.get('tooth_rbox_quantile_low', 0.05)),
                tooth_rbox_quantile_high=float(self.risk_query_cfg.get('tooth_rbox_quantile_high', 0.95)),
                tooth_rbox_expand_ratio=float(self.risk_query_cfg.get('tooth_rbox_expand_ratio', 1.05)),
                tooth_rbox_min_points=int(self.risk_query_cfg.get('tooth_rbox_min_points', 12)),
                tooth_rbox_max_points=int(self.risk_query_cfg.get('tooth_rbox_max_points', 512)),
                tooth_rbox_bbox_clip_margin=float(self.risk_query_cfg.get('tooth_rbox_bbox_clip_margin', 0.02)),
                tooth_expanded_box_ratio=float(self.risk_query_cfg.get('tooth_expanded_box_ratio', 1.1)),
                risk_gaussian_sigma_min=float(self.risk_query_cfg.get('risk_gaussian_sigma_min', 1.0)),
                risk_gaussian_sigma_scale=float(self.risk_query_cfg.get('risk_gaussian_sigma_scale', 0.5)),
                hard_miss_confidence_k=self.hard_miss_confidence_k,
                hard_miss_ema_momentum=self.hard_miss_ema_momentum,
                upper_tooth_types=self.risk_query_cfg.get('upper_tooth_types', None),
                lower_tooth_types=self.risk_query_cfg.get('lower_tooth_types', None),
            )
            self.encoder_prior_adapter = EncoderPriorAttentionAdapter(
                hidden_dim, act=mlp_act
            ) if self.use_score_only_encoder_adapter else None
            self.image_risk_head = MLP(hidden_dim, hidden_dim, 1, 2, act=mlp_act)
            # Learnable [stat_risk, hard_miss, image_risk] fusion weights.
            # softplus keeps weights positive; init reproduces fixed [0.5, 0.5, 1.0].
            self.risk_score_fusion = None
            self.risk_score_fusion_raw = nn.Parameter(
                _inv_softplus(torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32))
            )

            # Bias is important because the fused prior is non-negative.
            # Without bias, BCE logits would be >= 0 everywhere, making background hard to suppress.
            self.risk_score_bias = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
            self.lambda_raw = None
        else:
            self.tooth_prior_builder = None
            self.encoder_prior_adapter = None
            self.image_risk_head = None
            self.risk_score_fusion = None
            self.risk_score_fusion_raw = None
            self.risk_score_bias = None
            self.lambda_raw = None

        # decoder head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
          + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max+1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
          + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max+1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)])
        self.integral = Integral(self.reg_max)

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()


        self._reset_parameters(feat_channels)

    def set_epoch(self, epoch):
        self.current_epoch.fill_(float(epoch))

    def _prior_bias_warmup_scale(self) -> float:
        # #5 linear ramp min(1, epoch / warmup), multiplied by an explicit max strength.
        # This lets query selection keep using the prior while decoder attention bias is weakened
        # or disabled for clean ablations.
        strength = float(getattr(self, 'decoder_prior_bias_strength', 1.0))
        w = float(getattr(self, 'prior_bias_warmup_epochs', 0.0) or 0.0)
        if w <= 0.0:
            warmup = 1.0
        else:
            e = float(self.current_epoch.item())
            warmup = max(0.0, min(1.0, e / w))
        return strength * warmup

    def _lambda_warmup_factor(self):
        return 1.0

    def _lambdas(self):
        if self.lambda_raw is None:
            return torch.zeros((3,), device=self.current_epoch.device)
        return F.softplus(self.lambda_raw)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)
        if self.image_risk_head is not None:
            init.constant_(self.image_risk_head.layers[-1].weight, 0)
            init.constant_(self.image_risk_head.layers[-1].bias, 0)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask



    def _compute_image_risk_logits(self, output_memory: torch.Tensor, prior_dict: dict = None) -> torch.Tensor:
        if self.image_risk_head is None:
            return output_memory.new_zeros(output_memory.shape[:2])

        return self.image_risk_head(output_memory).squeeze(-1)

    def _risk_score_lambdas(self, device=None, dtype=None) -> torch.Tensor:
        # #4 learnable, strictly-positive fusion weights (softplus of the raw parameter).
        if getattr(self, 'risk_score_fusion_raw', None) is not None:
            return F.softplus(self.risk_score_fusion_raw).to(device=device, dtype=dtype)
        if self.risk_score_fusion is None:
            return torch.zeros((3,), device=device or self.current_epoch.device, dtype=dtype or torch.float32)
        return self.risk_score_fusion.to(device=device, dtype=dtype)

    def _risk_template_loss(self, image_risk_logits: torch.Tensor, prior_dict: dict) -> torch.Tensor:
        if image_risk_logits is None or prior_dict is None:
            return torch.zeros((), device=self.current_epoch.device)
        target = prior_dict.get('risk_template_target')
        mask = prior_dict.get('risk_template_mask')
        if target is None or mask is None or not mask.any():
            return image_risk_logits.sum() * 0.0
        logits = image_risk_logits.float()
        target = target.to(device=logits.device, dtype=logits.dtype).clamp(0, 1)
        mask = mask.to(device=logits.device, dtype=torch.bool)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = 0.25 * target + 0.75 * (1.0 - target)
        loss = alpha_t * (1.0 - p_t).pow(2.0) * ce
        return loss[mask].sum() / mask.sum().clamp_min(1).to(loss.dtype)


    def _risk_score_prior_loss(self, prior_dict: dict) -> torch.Tensor:
        if prior_dict is None:
            return torch.zeros((), device=self.current_epoch.device)

        logits = prior_dict.get('risk_score_prior_logits')
        target = prior_dict.get('risk_template_target')
        mask = prior_dict.get('risk_template_mask')

        if logits is None or target is None or mask is None:
            return torch.zeros((), device=self.current_epoch.device)

        logits = logits.float()
        target = target.to(device=logits.device, dtype=logits.dtype).clamp(0, 1)
        mask = mask.to(device=logits.device, dtype=torch.bool)

        # Important for DDP: keep a graph connection even if this batch has no valid mask.
        if not mask.any():
            return logits.sum() * 0.0

        ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = 0.25 * target + 0.75 * (1.0 - target)

        loss = alpha_t * (1.0 - p_t).pow(2.0) * ce
        return loss[mask].sum() / mask.sum().clamp_min(1).to(loss.dtype)


    def _decoder_prior_maps(self, prior_dict: dict, dtype: torch.dtype = None) -> torch.Tensor:
        if prior_dict is None:
            return None
        base = prior_dict.get('risk_prior_norm')
        if base is None:
            return None
        maps = torch.stack([
            prior_dict.get('risk_prior_norm', torch.zeros_like(base)),
            prior_dict.get('hard_prior', torch.zeros_like(base)),
            prior_dict.get('image_risk_prior', torch.zeros_like(base)),
            prior_dict.get('tooth_prior', torch.zeros_like(base)),
            prior_dict.get('context_prior', torch.zeros_like(base)),
        ], dim=-1)
        return maps.to(dtype=dtype) if dtype is not None else maps

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None,
                           prior_dict=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        score_memory = output_memory

        if self.risk_query_enabled and prior_dict is not None:
            image_risk_logits = self._compute_image_risk_logits(output_memory, prior_dict)

            # For query selection, use detached image risk to keep selection stable.
            image_risk_prior = torch.sigmoid(image_risk_logits.detach()).to(output_memory.dtype)

            # For auxiliary supervision, keep a differentiable image risk branch.
            image_risk_prior_train = torch.sigmoid(image_risk_logits).to(output_memory.dtype)

            prior_dict['image_risk_logits'] = image_risk_logits
            prior_dict['image_risk_prior'] = image_risk_prior

            lambdas = self._risk_score_lambdas(device=output_memory.device, dtype=output_memory.dtype)

            stat_prior = prior_dict.get('risk_prior_norm', torch.zeros_like(image_risk_prior)).to(output_memory.dtype)
            hard_prior = prior_dict.get('hard_prior', torch.zeros_like(image_risk_prior)).to(output_memory.dtype)

            # This is the actual prior used for hard top-k query selection.
            prior_dict['risk_score_prior'] = (
                lambdas[0] * stat_prior
                + lambdas[1] * hard_prior
                + lambdas[2] * image_risk_prior
            )

            # This is the differentiable version supervised by an auxiliary loss.
            risk_score_prior_train = (
                lambdas[0] * stat_prior
                + lambdas[1] * hard_prior
                + lambdas[2] * image_risk_prior_train
            )

            prior_dict['risk_score_prior_logits'] = (
                risk_score_prior_train
                + self.risk_score_bias.to(device=output_memory.device, dtype=output_memory.dtype)
            )

        if self.risk_query_enabled and prior_dict is not None and self.encoder_prior_adapter is not None:
            # #13 feed the per-type normalized risk prior (in [0, 1]) so the risk channel is on the
            # same scale as the 0/1 tooth/context masks; the raw risk_prior (~1e-3) would be drowned out.
            priors = torch.stack([
                prior_dict['risk_prior_norm'],
                prior_dict['tooth_inside_mask'].to(output_memory.dtype),
                prior_dict['context_mask'].to(output_memory.dtype),
            ], dim=-1).to(dtype=output_memory.dtype)
            score_memory = self.encoder_prior_adapter(output_memory, priors)

        enc_outputs_logits: torch.Tensor = self.enc_score_head(score_memory)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors, query_diag, query_groups, query_role_info = \
            self._select_topk(output_memory, enc_outputs_logits, anchors, self.num_queries, prior_dict=prior_dict)

        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
            if query_groups is not None:
                dn_groups = torch.full((query_groups.shape[0], denoising_bbox_unact.shape[1]), -1,
                                       device=query_groups.device, dtype=query_groups.dtype)
                query_groups = torch.cat([dn_groups, query_groups], dim=1)

        # #9 assemble a full-length tooth-local (u, v) tensor aligned with the decoder query order.
        # Denoising queries (prepended) carry zeros; matching queries take their selected (u, v).
        query_uv = None
        if query_role_info is not None and 'query_point_u' in query_role_info and 'query_point_v' in query_role_info:
            query_uv = torch.stack([query_role_info['query_point_u'], query_role_info['query_point_v']], dim=-1)
            if denoising_bbox_unact is not None:
                num_dn = int(denoising_bbox_unact.shape[1])
                dn_uv = torch.zeros((query_uv.shape[0], num_dn, 2), device=query_uv.device, dtype=query_uv.dtype)
                query_uv = torch.cat([dn_uv, query_uv], dim=1)

        # Local-u axis tensor for proximal contrast; DN queries carry zero fallback axes.
        query_u_axes = None
        if query_role_info is not None and 'query_point_u_axis_x' in query_role_info and 'query_point_u_axis_y' in query_role_info:
            query_u_axes = torch.stack([query_role_info['query_point_u_axis_x'], query_role_info['query_point_u_axis_y']], dim=-1)
            if denoising_bbox_unact is not None:
                num_dn = int(denoising_bbox_unact.shape[1])
                dn_axes = torch.zeros((query_u_axes.shape[0], num_dn, 2), device=query_u_axes.device, dtype=query_u_axes.dtype)
                query_u_axes = torch.cat([dn_axes, query_u_axes], dim=1)

        # Optimization 3: assemble a full-length tooth-type tensor; DN queries carry -1 (no bias).
        query_type = None
        if query_role_info is not None and 'query_point_type' in query_role_info:
            query_type = query_role_info['query_point_type']
            if denoising_bbox_unact is not None:
                num_dn = int(denoising_bbox_unact.shape[1])
                dn_type = torch.full((query_type.shape[0], num_dn), -1, device=query_type.device, dtype=query_type.dtype)
                query_type = torch.cat([dn_type, query_type], dim=1)

        # #4 full-length per-query anatomical midline x (normalized), for contralateral mirroring.
        # DN queries (prepended) carry 0.5 (image center) as a harmless fallback.
        query_midline_x = None
        if query_role_info is not None and 'query_point_midline_x' in query_role_info:
            query_midline_x = query_role_info['query_point_midline_x']
            if denoising_bbox_unact is not None:
                num_dn = int(denoising_bbox_unact.shape[1])
                dn_mid = torch.full((query_midline_x.shape[0], num_dn), 0.5,
                                    device=query_midline_x.device, dtype=query_midline_x.dtype)
                query_midline_x = torch.cat([dn_mid, query_midline_x], dim=1)

        # #5 full-length per-query tooth id (>=0 inside-tooth/context, -1 otherwise), for intra-tooth
        # self-attention.  DN queries carry -1 so they never join any tooth block.
        query_tooth = None
        if query_role_info is not None and 'query_point_tooth' in query_role_info:
            query_tooth = query_role_info['query_point_tooth']
            if denoising_bbox_unact is not None:
                num_dn = int(denoising_bbox_unact.shape[1])
                dn_tooth = torch.full((query_tooth.shape[0], num_dn), -1, device=query_tooth.device, dtype=query_tooth.dtype)
                query_tooth = torch.cat([dn_tooth, query_tooth], dim=1)

        return (content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, query_diag,
                query_groups, query_role_info, query_uv, query_u_axes, query_type, query_midline_x, query_tooth)

    def _base_encoder_score(self, outputs_logits: torch.Tensor) -> torch.Tensor:
        if outputs_logits.shape[-1] == 1:
            return outputs_logits.squeeze(-1)
        idx = max(0, min(int(self.caries_class_index), outputs_logits.shape[-1] - 1))
        return outputs_logits[..., idx]

    def _select_in_mask(self, score: torch.Tensor, mask: torch.Tensor, selected_mask: torch.Tensor, k: int):
        bs, length = score.shape
        selected_indices, duplicate_counts, fill_counts = [], [], []
        for b in range(bs):
            region = mask[b].bool()
            prev = selected_mask[b]
            valid = region & (~prev)
            valid_count = min(int(k), int(valid.sum().item()))
            idx_parts = []
            if valid_count > 0:
                idx_parts.append(torch.topk(score[b].masked_fill(~valid, -torch.inf), valid_count).indices)
            fill = int(k) - valid_count
            if fill > 0:
                # Still fill only inside the group's own region; allow repeated/previously selected points if needed.
                fill_count = min(fill, int(region.sum().item()))
                if fill_count > 0:
                    idx_parts.append(torch.topk(score[b].masked_fill(~region, -torch.inf), fill_count).indices)
                fill -= fill_count
            idx = torch.cat(idx_parts, dim=0) if idx_parts else torch.zeros((0,), device=score.device, dtype=torch.long)
            if idx.numel() < k:
                pad = idx[:1].repeat(k - idx.numel()) if idx.numel() > 0 else torch.zeros((k - idx.numel(),), device=score.device, dtype=torch.long)
                idx = torch.cat([idx, pad], dim=0)
            selected_indices.append(idx[:k])
            duplicate_counts.append(prev[idx[:k]].sum().to(score.dtype))
            fill_counts.append(score.new_tensor(float(max(0, int(k) - valid_count))))
        return torch.stack(selected_indices, dim=0), torch.stack(duplicate_counts), torch.stack(fill_counts)

    def _largest_remainder_allocation(self, total: int, weights: torch.Tensor) -> torch.Tensor:
        if total <= 0 or weights.numel() == 0:
            return torch.zeros_like(weights, dtype=torch.long)
        weights = weights.to(torch.float32).clamp_min(0)
        if float(weights.sum()) <= 0:
            weights = torch.ones_like(weights)
        raw = weights / weights.sum() * int(total)
        base = raw.floor().long()
        remain = int(total - int(base.sum().item()))
        if remain > 0:
            frac = raw - base.to(raw.dtype)
            order = frac.argsort(descending=True)
            base[order[:remain]] += 1
        return base

    def _select_tooth_queries(self, base_score: torch.Tensor, prior_dict: dict, selected_mask: torch.Tensor, k: int):
        bs, length = base_score.shape
        point_tooth = prior_dict['point_tooth']
        point_type = prior_dict['point_type']
        tooth_mask = prior_dict['tooth_inside_mask'].bool()
        p_eff = prior_dict.get('p_eff')
        selected_indices = []
        dup_counts, fill_counts = [], []

        for b in range(bs):
            prev = selected_mask[b]
            chosen = []
            local_selected = torch.zeros((length,), device=base_score.device, dtype=torch.bool)
            tooth_ids = torch.unique(point_tooth[b][point_tooth[b] >= 0])
            # Stage 1: per-tooth minimum coverage.
            if tooth_ids.numel() > 0 and self.min_tooth_queries_per_tooth > 0:
                # Round-robin protects against abnormal cases where min coverage exceeds tooth budget.
                for round_idx in range(self.min_tooth_queries_per_tooth):
                    if len(chosen) >= k:
                        break
                    for tid in tooth_ids.tolist():
                        if len(chosen) >= k:
                            break
                        region = tooth_mask[b] & (point_tooth[b] == int(tid)) & (~prev) & (~local_selected)
                        if region.sum() == 0:
                            region = tooth_mask[b] & (point_tooth[b] == int(tid)) & (~local_selected)
                        if region.sum() == 0:
                            region = tooth_mask[b] & (point_tooth[b] == int(tid))
                        if region.sum() > 0:
                            idx = torch.topk(base_score[b].masked_fill(~region, -torch.inf), 1).indices
                            chosen.append(idx)
                            local_selected[idx] = True

            remaining = max(0, k - len(chosen))
            if remaining > 0:
                type_ids = torch.unique(point_type[b][point_type[b] >= 0])
                weights = []
                valid_types = []
                for t in type_ids.tolist():
                    n_teeth = torch.unique(point_tooth[b][(point_type[b] == int(t)) & (point_tooth[b] >= 0)]).numel()
                    if n_teeth <= 0:
                        continue
                    p_val = p_eff[int(t)].detach().to(base_score.device) if p_eff is not None and int(t) < p_eff.numel() else base_score.new_tensor(1.0)
                    weights.append(float(n_teeth) * torch.pow(p_val + self.tooth_extra_eps, self.tooth_extra_gamma))
                    valid_types.append(int(t))
                if valid_types:
                    weights = torch.stack([w if torch.is_tensor(w) else base_score.new_tensor(float(w)) for w in weights]).to(base_score.device)
                    alloc = self._largest_remainder_allocation(remaining, weights)
                    for t, kk in zip(valid_types, alloc.tolist()):
                        if kk <= 0:
                            continue
                        region = tooth_mask[b] & (point_type[b] == int(t)) & (~prev) & (~local_selected)
                        count = min(int(kk), int(region.sum().item()))
                        if count > 0:
                            idx = torch.topk(base_score[b].masked_fill(~region, -torch.inf), count).indices
                            chosen.append(idx)
                            local_selected[idx] = True
                        fill = int(kk) - count
                        if fill > 0:
                            region2 = tooth_mask[b] & (point_type[b] == int(t))
                            count2 = min(fill, int(region2.sum().item()))
                            if count2 > 0:
                                idx = torch.topk(base_score[b].masked_fill(~region2, -torch.inf), count2).indices
                                chosen.append(idx)
                                local_selected[idx] = True

            idx = torch.cat(chosen, dim=0) if chosen else torch.zeros((0,), device=base_score.device, dtype=torch.long)
            if idx.numel() < k:
                region = tooth_mask[b]
                count = min(k - idx.numel(), int(region.sum().item()))
                if count > 0:
                    idx = torch.cat([idx, torch.topk(base_score[b].masked_fill(~region, -torch.inf), count).indices], dim=0)
            if idx.numel() < k:
                pad = idx[:1].repeat(k - idx.numel()) if idx.numel() > 0 else torch.zeros((k - idx.numel(),), device=base_score.device, dtype=torch.long)
                idx = torch.cat([idx, pad], dim=0)
            idx = idx[:k]
            selected_indices.append(idx)
            dup_counts.append(prev[idx].sum().to(base_score.dtype))
            fill_counts.append(base_score.new_tensor(float(max(0, k - int((tooth_mask[b] & (~prev)).sum().item())))))
        return torch.stack(selected_indices, dim=0), torch.stack(dup_counts), torch.stack(fill_counts)

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor,
                     topk: int, prior_dict=None):
        query_diag = None
        query_groups = None
        query_role_info = None
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
            query_groups = torch.zeros_like(topk_ind)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
            query_groups = torch.zeros_like(topk_ind)

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
            query_groups = torch.zeros_like(topk_ind)

        elif self.query_select_method == 'risk_group':
            base_score = self._base_encoder_score(outputs_logits)
            if prior_dict is None:
                prior_dict = {
                    'risk_score_prior': torch.zeros_like(base_score),
                    'tooth_inside_mask': torch.ones_like(base_score, dtype=torch.bool),
                    'context_mask': torch.ones_like(base_score, dtype=torch.bool),
                    'global_mask': torch.ones_like(base_score, dtype=torch.bool),
                    'point_tooth': torch.full_like(base_score, -1, dtype=torch.long),
                    'point_type': torch.full_like(base_score, -1, dtype=torch.long),
                }
            selected_mask = torch.zeros_like(base_score, dtype=torch.bool)
            all_indices, all_groups = [], []
            diag = {}

            risk_score = base_score + prior_dict.get('risk_score_prior', torch.zeros_like(base_score)).to(base_score.dtype)
            risk_idx, dup, fill = self._select_in_mask(risk_score, prior_dict['tooth_inside_mask'], selected_mask, self.num_risk_queries)
            selected_mask.scatter_(1, risk_idx, True)
            all_indices.append(risk_idx); all_groups.append(torch.full_like(risk_idx, 1))
            diag['risk_duplicate_count'] = dup.detach(); diag['risk_fill_count'] = fill.detach(); diag['risk_requested'] = self.num_risk_queries

            tooth_idx, dup, fill = self._select_tooth_queries(base_score, prior_dict, selected_mask, self.num_tooth_queries)
            selected_mask.scatter_(1, tooth_idx, True)
            all_indices.append(tooth_idx); all_groups.append(torch.full_like(tooth_idx, 2))
            diag['tooth_duplicate_count'] = dup.detach(); diag['tooth_fill_count'] = fill.detach(); diag['tooth_requested'] = self.num_tooth_queries

            context_idx, dup, fill = self._select_in_mask(base_score, prior_dict['context_mask'], selected_mask, self.num_context_queries)
            selected_mask.scatter_(1, context_idx, True)
            all_indices.append(context_idx); all_groups.append(torch.full_like(context_idx, 3))
            diag['context_duplicate_count'] = dup.detach(); diag['context_fill_count'] = fill.detach(); diag['context_requested'] = self.num_context_queries

            global_idx, dup, fill = self._select_in_mask(base_score, prior_dict['global_mask'], selected_mask, self.num_global_queries)
            selected_mask.scatter_(1, global_idx, True)
            all_indices.append(global_idx); all_groups.append(torch.full_like(global_idx, 0))
            diag['global_duplicate_count'] = dup.detach(); diag['global_fill_count'] = fill.detach(); diag['global_requested'] = self.num_global_queries

            topk_ind = torch.cat(all_indices, dim=1)
            query_groups = torch.cat(all_groups, dim=1)
            def _gather_prior(name, default, is_long=False):
                src = prior_dict.get(name, default)
                gathered = src.gather(dim=1, index=topk_ind)
                return gathered.long() if is_long else gathered
            query_role_info = {
                'query_groups': query_groups.detach(),
                'query_point_tooth': _gather_prior('point_tooth', torch.full_like(base_score, -1, dtype=torch.long), is_long=True).detach(),
                'query_point_type': _gather_prior('point_type', torch.full_like(base_score, -1, dtype=torch.long), is_long=True).detach(),
                'query_point_u': _gather_prior('point_u', torch.zeros_like(base_score)).detach(),
                'query_point_v': _gather_prior('point_v', torch.zeros_like(base_score)).detach(),
                'query_point_u_axis_x': _gather_prior('point_u_axis_x', torch.zeros_like(base_score)).detach(),
                'query_point_u_axis_y': _gather_prior('point_u_axis_y', torch.zeros_like(base_score)).detach(),
                'query_point_midline_x': _gather_prior('point_midline_x', torch.zeros_like(base_score)).detach(),
                'query_risk_prior': _gather_prior('risk_prior_norm', torch.zeros_like(base_score)).detach(),
                'query_hard_prior': _gather_prior('hard_prior', torch.zeros_like(base_score)).detach(),
                'query_image_risk_prior': _gather_prior('image_risk_prior', torch.zeros_like(base_score)).detach(),
            }
            if not self.training:
                prior_channels = torch.stack([
                    prior_dict.get('risk_prior', torch.zeros_like(base_score)),
                    prior_dict.get('tooth_prior', torch.zeros_like(base_score)),
                    prior_dict.get('context_prior', torch.zeros_like(base_score)),
                ], dim=-1)
                selected_priors = prior_channels.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, 3))
                query_diag = {'selected_groups': query_groups.detach(),
                              'selected_priors': selected_priors.detach(), **diag}

        topk_anchors = outputs_anchors_unact.gather(dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))

        topk_logits = outputs_logits.gather(dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None

        topk_memory = memory.gather(dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        if query_diag is not None:
            query_diag['selected_anchor_boxes'] = F.sigmoid(topk_anchors.detach())

        return topk_memory, topk_logits, topk_anchors, query_diag, query_groups, query_role_info

    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)
        prior_dict = None
        if self.risk_query_enabled and self.tooth_prior_builder is not None and targets is not None:
            prior_dict = self.tooth_prior_builder(targets, spatial_shapes, memory.device, torch.float32)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                    )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        (init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, query_diag,
         query_groups, query_role_info, query_uv, query_u_axes, query_type, query_midline_x, query_tooth) = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact, prior_dict=prior_dict)
        decoder_prior_maps = self._decoder_prior_maps(prior_dict, dtype=memory.dtype)

        # decoder
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            prior_maps=decoder_prior_maps,
            query_groups=query_groups,
            prior_bias_scale=self._prior_bias_warmup_scale(),
            query_uv=query_uv,
            query_u_axes=query_u_axes,
            query_type=query_type,
            query_midline_x=query_midline_x,
            query_tooth=query_tooth)

        if self.training and dn_meta is not None:
            # the output from the first decoder layer, only one
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)

            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)

            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)


        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale}
            if prior_dict is not None and 'image_risk_logits' in prior_dict:
                out['loss_risk_template'] = self._risk_template_loss(prior_dict['image_risk_logits'], prior_dict)
                out['loss_risk_score_prior'] = self._risk_score_prior_loss(prior_dict)

            # #11 anchor the semi-learnable risk-map residuals toward the frozen statistics.
            if self.tooth_prior_builder is not None:
                out['loss_risk_reg'] = self.tooth_prior_builder.risk_map.residual_reg_loss()
            # #6 per-GT hard-miss positive-loss multipliers (no-op until beta>0 and stats exist).
            if (self.tooth_prior_builder is not None and targets is not None
                    and float(self.hard_miss_loss_beta) > 0.0
                    and bool(self.tooth_prior_builder.risk_map.risk_stats_initialized.item())):
                h_stat = self.tooth_prior_builder.risk_map.h_stat
                expand_ratio = float(self.tooth_prior_builder.tooth_expanded_box_ratio)
                num_types = int(self.tooth_prior_builder.num_tooth_types)
                gt_hard_weights = []
                for t in targets:
                    a = float(t['tooth_rbox_aspect'].reshape(-1)[0].item()) if 'tooth_rbox_aspect' in t else 1.0
                    gt_hard_weights.append(
                        gt_hard_weights_for_target(t, h_stat, image_aspect=a,
                                                   beta=float(self.hard_miss_loss_beta),
                                                   expand_ratio=expand_ratio, num_types=num_types))
                out['gt_hard_weights'] = gt_hard_weights
            if query_role_info is not None:
                out.update(query_role_info)
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
            if query_diag is not None:
                out['query_diag'] = query_diag

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
                                                     out_corners[-1], out_logits[-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs,
                                                        dn_out_corners[-1], dn_out_logits[-1])
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]


    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                       teacher_corners=None, teacher_logits=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
                     'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]
