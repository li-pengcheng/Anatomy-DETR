"""
Stage 1+ tooth-aware prior construction, score-only adapter, and auxiliary risk-template loss.

This version supports tooth category ids from the prompt json and optional segmentation-driven
rotated boxes.  Segmentation masks are not used as hard pixel gates; they are used to estimate a
robust tooth pose/rotated box after data augmentation, so the risk template can be sampled in a
local coordinate system aligned to the tooth.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .tooth_utils import cxcywh_to_xyxy, expand_xyxy_boxes


DEFAULT_UPPER_TOOTH_TYPES = (0, 2, 4)
DEFAULT_LOWER_TOOTH_TYPES = (1, 3, 5)


def _normalize_type_tuple(values, default):
    if values is None:
        values = default
    if isinstance(values, str):
        values = [v.strip() for v in values.split(',') if v.strip() != '']
    return tuple(int(v) for v in values)


def _type_membership(types: torch.Tensor, values) -> torch.Tensor:
    values = _normalize_type_tuple(values, ())
    mask = torch.zeros_like(types, dtype=torch.bool)
    for t in values:
        mask |= types == int(t)
    return mask


# --- Aspect correction note -------------------------------------------------
# Panoramic frames are far from square (W / H ~ 2, and other datasets are even more
# extreme).  The network sees square pixels, so any tooth pose must be estimated in an
# isotropic frame.  We therefore run PCA / inside-tests / uv mapping in an "aspect
# corrected" frame where the normalized x coordinate is multiplied by ``a = W / H``.
# Rotated boxes store the center in plain normalized [0, 1] coordinates, while (w, h, angle)
# live in this corrected frame.  Every consumer maps a query point (x, y) -> (a * x, y)
# before rotating, which makes the recovered angle match the true pixel-space angle and
# keeps q_stat construction and inference perfectly self-consistent.  ``image_aspect``
# defaults to 1.0 (no correction) so any legacy call path degrades gracefully.


def _aspect_from_target(target: dict, default: float = 1.0) -> float:
    """Read the aspect factor (W / H of the network frame) stored by the rbox transform."""
    if isinstance(target, dict) and 'tooth_rbox_aspect' in target:
        try:
            val = float(target['tooth_rbox_aspect'].reshape(-1)[0].item())
            return val if val > 0 else default
        except Exception:
            return default
    return default


def _aspect_from_spatial_shapes(spatial_shapes, default: float = 1.0) -> float:
    """Image aspect W / H inferred from the (h, w) of any feature level."""
    try:
        h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
        if h > 0 and w > 0:
            return float(w) / float(h)
    except Exception:
        pass
    return default


def _symeig2x2(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Closed-form eigendecomposition of the 2x2 symmetric matrix [[a, b], [b, c]].

    Returns (lambda_major, lambda_minor, major_axis[2]).  This avoids the LAPACK overhead of
    ``torch.linalg.eigh`` on tiny per-tooth covariances and matches it to fp precision (see the
    numpy validation accompanying this change).
    """
    tr = a + c
    diff = torch.sqrt(((a - c) * 0.5) ** 2 + b * b)
    l_major = tr * 0.5 + diff
    l_minor = tr * 0.5 - diff
    # Eigenvector for the major eigenvalue.  When b ~ 0 the matrix is already diagonal.
    vx = l_major - c
    vy = b
    near_diag = b.abs() <= 1e-12
    vx = torch.where(near_diag, torch.where(a >= c, torch.ones_like(a), torch.zeros_like(a)), vx)
    vy = torch.where(near_diag, torch.where(a >= c, torch.zeros_like(a), torch.ones_like(a)), vy)
    norm = torch.sqrt(vx * vx + vy * vy).clamp_min(1e-12)
    return l_major, l_minor, torch.stack([vx / norm, vy / norm])


def _is_upper_type(type_id: int, upper_tooth_types=None) -> bool:
    return int(type_id) in _normalize_type_tuple(upper_tooth_types, DEFAULT_UPPER_TOOTH_TYPES)


def _is_lower_type(type_id: int, lower_tooth_types=None) -> bool:
    return int(type_id) in _normalize_type_tuple(lower_tooth_types, DEFAULT_LOWER_TOOTH_TYPES)


class EncoderPriorAttentionAdapter(nn.Module):
    """Lightweight score-only adapter. Final projection is zero-initialized."""

    def __init__(self, hidden_dim: int, prior_dim: int = 3, act: str = "silu"):
        super().__init__()
        self.memory_proj = nn.Linear(hidden_dim, hidden_dim)
        self.prior_proj = nn.Linear(prior_dim, hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU() if act == "silu" else nn.ReLU(inplace=True)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, output_memory: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        if priors is None:
            return output_memory
        delta = self.act(self.memory_proj(output_memory) + self.prior_proj(priors.to(output_memory.dtype)))
        return output_memory + self.final_proj(delta)


def _axis_aligned_rboxes_from_xyxy(boxes_xyxy: torch.Tensor, tooth_types: Optional[torch.Tensor] = None,
                                   image_aspect: float = 1.0,
                                   upper_tooth_types=None, lower_tooth_types=None) -> torch.Tensor:
    """Fallback rotated boxes [cx, cy, w, h, angle] from axis-aligned boxes.

    The fallback angle is type-aware so local v keeps the same crown->root convention:
    upper teeth point to smaller y, lower teeth point to larger y.

    Convention (see module note on aspect correction): center (cx, cy) is stored in
    normalized [0, 1] image coordinates, while (w, h, angle) live in the aspect-corrected
    isotropic frame where the x axis is multiplied by ``image_aspect`` = W / H.  For an
    axis-aligned box the angle is 0 / pi so only the width carries the aspect factor; the
    factor cancels out in every consumer, so axis-aligned boxes are aspect-invariant.
    """
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((0, 5))
    a = float(image_aspect) if float(image_aspect) > 0 else 1.0
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    angle = torch.zeros_like(x1)
    if tooth_types is not None and tooth_types.numel() == boxes_xyxy.shape[0]:
        types = tooth_types.to(device=boxes_xyxy.device).long().reshape(-1)
        upper = _type_membership(types, _normalize_type_tuple(upper_tooth_types, DEFAULT_UPPER_TOOTH_TYPES))
        angle = torch.where(upper, angle.new_full(angle.shape, torch.pi), angle)
    return torch.stack([
        (x1 + x2) * 0.5,
        (y1 + y2) * 0.5,
        ((x2 - x1) * a).clamp(min=1e-6),
        (y2 - y1).clamp(min=1e-6),
        angle,
    ], dim=-1)


def _orient_root_axis(axis: torch.Tensor, type_id: Optional[int],
                      upper_tooth_types=None, lower_tooth_types=None) -> torch.Tensor:
    """Orient the tooth long axis so positive local v means crown -> root.

    upper_tooth_types / lower_tooth_types are configurable because the 6-class tooth
    taxonomy is project-specific.  A wrong hard-coded mapping silently flips the
    crown-root v coordinate and corrupts the risk/hard-miss maps.
    """
    if type_id is None:
        return axis if axis[1] >= 0 else -axis
    if _is_upper_type(type_id, upper_tooth_types):
        return axis if axis[1] <= 0 else -axis
    if _is_lower_type(type_id, lower_tooth_types):
        return axis if axis[1] >= 0 else -axis
    return axis if axis[1] >= 0 else -axis


def _u_axis_from_root_axis(v_axis: torch.Tensor) -> torch.Tensor:
    """Return the local u axis whose +90-degree perpendicular is the root-oriented v axis."""
    return torch.stack([v_axis[1], -v_axis[0]])


def _rboxes_from_masks(
    masks: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    tooth_types: Optional[torch.Tensor] = None,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    expand_ratio: float = 1.05,
    min_points: int = 12,
    max_points: int = 512,
    bbox_clip_margin: float = 0.02,
    image_aspect: float = 1.0,
    upper_tooth_types=None,
    lower_tooth_types=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate robust normalized rotated boxes from transformed tooth masks.

    The pose (angle / extents) is estimated in the aspect-corrected isotropic frame (x scaled by
    ``image_aspect``), so the recovered angle matches the true pixel-space angle regardless of how
    far the panoramic frame departs from 1:1.  The returned rbox stores the center in normalized
    [0, 1] coordinates and (w, h, angle) in the corrected frame.  A mask only needs to cover the
    main tooth body; noisy boundaries are suppressed by percentile clipping.  Invalid / empty masks
    fall back to the axis-aligned prompt box and are flagged invalid.

    Speed: a closed-form 2x2 eigendecomposition replaces ``torch.linalg.eigh``, points are
    subsampled to ``max_points`` (default 512, ample for pose), and the low/high percentiles of both
    axes are taken in a single batched ``torch.quantile`` call.
    """
    a = float(image_aspect) if float(image_aspect) > 0 else 1.0
    fallback = _axis_aligned_rboxes_from_xyxy(boxes_xyxy, tooth_types, image_aspect=a,
                                             upper_tooth_types=upper_tooth_types,
                                             lower_tooth_types=lower_tooth_types)
    valid = torch.zeros((boxes_xyxy.shape[0],), device=boxes_xyxy.device, dtype=torch.bool)
    if masks is None or boxes_xyxy.numel() == 0:
        return fallback, valid

    raw_masks = masks.as_subclass(torch.Tensor) if hasattr(masks, 'as_subclass') else masks
    if raw_masks.ndim != 3 or raw_masks.shape[0] != boxes_xyxy.shape[0]:
        return fallback, valid

    raw_masks = raw_masks.to(device=boxes_xyxy.device)
    height, width = int(raw_masks.shape[-2]), int(raw_masks.shape[-1])
    if height <= 0 or width <= 0:
        return fallback, valid

    rboxes = fallback.clone()
    q_low = float(quantile_low)
    q_high = float(quantile_high)
    q_low = max(0.0, min(q_low, 0.49))
    q_high = max(q_low + 1e-3, min(q_high, 1.0))
    expand_ratio = max(float(expand_ratio), 1.0)
    q_levels = torch.tensor([q_low, q_high], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)

    for i in range(raw_masks.shape[0]):
        ys, xs = (raw_masks[i] > 0).nonzero(as_tuple=True)
        if xs.numel() < int(min_points):
            continue

        x1, y1, x2, y2 = boxes_xyxy[i]
        # Clip extreme mask outliers to the prompt bbox with a small margin.
        x_norm = (xs.to(boxes_xyxy.dtype) + 0.5) / max(width, 1)
        y_norm = (ys.to(boxes_xyxy.dtype) + 0.5) / max(height, 1)
        keep = (x_norm >= (x1 - bbox_clip_margin).clamp(0, 1)) & \
               (x_norm <= (x2 + bbox_clip_margin).clamp(0, 1)) & \
               (y_norm >= (y1 - bbox_clip_margin).clamp(0, 1)) & \
               (y_norm <= (y2 + bbox_clip_margin).clamp(0, 1))
        x_norm, y_norm = x_norm[keep], y_norm[keep]
        if x_norm.numel() < int(min_points):
            continue

        if x_norm.numel() > int(max_points):
            # Deterministic subsampling keeps the operation bounded without adding randomness.
            idx = torch.linspace(0, x_norm.numel() - 1, int(max_points), device=x_norm.device).long()
            x_norm, y_norm = x_norm[idx], y_norm[idx]

        # Work in the aspect-corrected isotropic frame: X = a * x_norm, Y = y_norm.
        pts = torch.stack([x_norm * a, y_norm], dim=1)
        mean = pts.mean(dim=0)
        centered = pts - mean
        n = max(int(pts.shape[0]) - 1, 1)
        cxx = (centered[:, 0] * centered[:, 0]).sum() / n
        cxy = (centered[:, 0] * centered[:, 1]).sum() / n
        cyy = (centered[:, 1] * centered[:, 1]).sum() / n
        l_major, _l_minor, major = _symeig2x2(cxx, cxy, cyy)
        if not torch.isfinite(l_major) or float(l_major) <= 1e-12:
            continue

        type_id = None
        if tooth_types is not None and tooth_types.numel() == boxes_xyxy.shape[0]:
            type_id = int(tooth_types[i].item())

        # PCA major axis is treated as the anatomical crown-root axis (local v).
        # Its sign is resolved from upper/lower tooth type so v=0 means crown/occlusal
        # and v=1 means root/apical.  The perpendicular axis is local u.
        axis_v = _orient_root_axis(major, type_id, upper_tooth_types=upper_tooth_types, lower_tooth_types=lower_tooth_types)
        axis_u = _u_axis_from_root_axis(axis_v)
        proj_u = centered.matmul(axis_u)
        proj_v = centered.matmul(axis_v)

        qu = torch.quantile(proj_u, q_levels)
        qv = torch.quantile(proj_v, q_levels)
        lo_u, hi_u = qu[0], qu[1]
        lo_v, hi_v = qv[0], qv[1]
        rw = (hi_u - lo_u).clamp(min=1e-6) * expand_ratio
        rh = (hi_v - lo_v).clamp(min=1e-6) * expand_ratio
        if not torch.isfinite(rw + rh) or float(rw) <= 1e-6 or float(rh) <= 1e-6:
            continue

        center = mean + axis_u * ((lo_u + hi_u) * 0.5) + axis_v * ((lo_v + hi_v) * 0.5)
        # center is in corrected frame; convert x back to plain normalized coordinates.
        angle = torch.atan2(axis_u[1], axis_u[0])
        rboxes[i] = torch.stack([
            (center[0] / a).clamp(0, 1),
            center[1].clamp(0, 1),
            rw,
            rh,
            angle,
        ])
        valid[i] = True

    return rboxes, valid


def _rbox_to_corners(rboxes: torch.Tensor, image_aspect: float = 1.0) -> torch.Tensor:
    """Return the 4 corners [N, 4, 2] (normalized image coords) of each rbox.

    rboxes store center (cx, cy) in normalized [0, 1] and (w, h, angle) in the aspect-corrected
    isotropic frame (x scaled by ``a = W / H``), consistent with every consumer in this module.
    Corner order is (-u,-v), (+u,-v), (+u,+v), (-u,+v).
    """
    a = float(image_aspect) if float(image_aspect) > 0 else 1.0
    if rboxes.numel() == 0:
        return rboxes.new_zeros((0, 4, 2))
    cx, cy, rw, rh, angle = rboxes.unbind(-1)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    u = torch.stack([cos_a, sin_a], dim=-1)      # [N, 2] corrected-frame unit axes
    v = torch.stack([-sin_a, cos_a], dim=-1)
    center = torch.stack([cx * a, cy], dim=-1)   # [N, 2] corrected frame
    out = []
    for su, sv in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        p = center + (su * rw)[:, None] * u + (sv * rh)[:, None] * v   # corrected frame
        p = torch.stack([p[:, 0] / a, p[:, 1]], dim=-1)               # back to normalized
        out.append(p)
    return torch.stack(out, dim=1)               # [N, 4, 2]


def _fit_rbox_from_corners(corners_norm: torch.Tensor,
                           tooth_types: Optional[torch.Tensor] = None,
                           image_aspect: float = 1.0,
                           upper_tooth_types=None, lower_tooth_types=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form rbox fit from 4 per-tooth corner points, in the aspect-corrected frame.

    ``corners_norm``: [N, 4, 2] normalized image coords (already carried through every geometric
    augmentation as tiny boxes).  Returns rboxes [N, 5] in the same (center-normalized,
    (w,h,angle)-corrected) convention every consumer expects, plus a validity flag [N].

    This replaces the heavy per-tooth mask PCA at training time: with exactly 4 clean points the
    covariance eigendecomposition and the extent min/max are O(1) per tooth.  Under an anisotropic
    resize a rotated rectangle becomes a parallelogram; the PCA fit returns the tight axis-aligned
    (in corrected-frame) bounding rbox, which stays consistent because consumers test inside-ness in
    the very same corrected frame (validated numerically alongside this change).
    """
    a = float(image_aspect) if float(image_aspect) > 0 else 1.0
    n = int(corners_norm.shape[0])
    rboxes = corners_norm.new_zeros((n, 5))
    valid = torch.zeros((n,), device=corners_norm.device, dtype=torch.bool)
    if n == 0:
        return rboxes, valid
    types = None
    if tooth_types is not None and tooth_types.numel() == n:
        types = tooth_types.to(device=corners_norm.device).long().reshape(-1)
    for i in range(n):
        pts = corners_norm[i].to(torch.float32)                 # [4, 2] normalized
        P = torch.stack([pts[:, 0] * a, pts[:, 1]], dim=1)      # corrected frame
        mean = P.mean(dim=0)
        X = P - mean
        nrm = max(int(P.shape[0]) - 1, 1)
        cxx = (X[:, 0] * X[:, 0]).sum() / nrm
        cxy = (X[:, 0] * X[:, 1]).sum() / nrm
        cyy = (X[:, 1] * X[:, 1]).sum() / nrm
        l_major, _l_minor, major = _symeig2x2(cxx, cxy, cyy)
        if not torch.isfinite(l_major) or float(l_major) <= 1e-12:
            continue
        type_id = int(types[i].item()) if types is not None else None
        axis_v = _orient_root_axis(major, type_id, upper_tooth_types=upper_tooth_types, lower_tooth_types=lower_tooth_types)
        axis_u = _u_axis_from_root_axis(axis_v)
        proj_u = X.matmul(axis_u)
        proj_v = X.matmul(axis_v)
        lo_u, hi_u = proj_u.min(), proj_u.max()
        lo_v, hi_v = proj_v.min(), proj_v.max()
        rw = (hi_u - lo_u).clamp(min=1e-6)
        rh = (hi_v - lo_v).clamp(min=1e-6)
        if not torch.isfinite(rw + rh) or float(rw) <= 1e-6 or float(rh) <= 1e-6:
            continue
        center = mean + axis_u * ((lo_u + hi_u) * 0.5) + axis_v * ((lo_v + hi_v) * 0.5)
        angle = torch.atan2(axis_u[1], axis_u[0])
        rboxes[i] = torch.stack([(center[0] / a).clamp(0, 1), center[1].clamp(0, 1), rw, rh, angle])
        valid[i] = True
    return rboxes, valid


def _rotated_local_coords(x: torch.Tensor, y: torch.Tensor, rboxes: torch.Tensor,
                          image_aspect: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return local x/y offsets for all boxes and points in the aspect-corrected frame."""
    a = float(image_aspect) if float(image_aspect) > 0 else 1.0
    cx, cy, _, _, angle = rboxes.unbind(-1)
    cos_a = torch.cos(angle)[:, None]
    sin_a = torch.sin(angle)[:, None]
    dx = (x[None] - cx[:, None]) * a
    dy = y[None] - cy[:, None]
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return local_x, local_y


def _rotated_inside(x: torch.Tensor, y: torch.Tensor, rboxes: torch.Tensor,
                    image_aspect: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_x, local_y = _rotated_local_coords(x, y, rboxes, image_aspect=image_aspect)
    rw = rboxes[:, 2].clamp(min=1e-6)[:, None]
    rh = rboxes[:, 3].clamp(min=1e-6)[:, None]
    inside = (local_x.abs() <= rw * 0.5) & (local_y.abs() <= rh * 0.5)
    return inside, local_x, local_y


def _mesial_u_flip(rboxes: torch.Tensor, tooth_midline_boxes: Optional[torch.Tensor],
                   midline_threshold: float = 0.01) -> torch.Tensor:
    """Return per-tooth flags indicating whether raw local u should be flipped.

    u=1 is defined as the mesial side.  The source-image midline is carried by
    tooth_midline_boxes through all geometric transforms, so this stays valid under
    Mosaic and horizontal flip.  Teeth very close to the midline are left unflipped.
    """
    flip = torch.zeros((rboxes.shape[0],), device=rboxes.device, dtype=torch.bool)
    if tooth_midline_boxes is None or rboxes.numel() == 0:
        return flip
    mids = tooth_midline_boxes.as_subclass(torch.Tensor) if hasattr(tooth_midline_boxes, 'as_subclass') else tooth_midline_boxes
    if mids.ndim != 2 or mids.shape[0] != rboxes.shape[0] or mids.shape[1] < 1:
        return flip
    mids = mids.to(device=rboxes.device, dtype=rboxes.dtype).reshape(-1, 4)
    mid_x = mids[:, 0]
    tooth_x = rboxes[:, 0]
    near_midline = (tooth_x - mid_x).abs() < float(midline_threshold)
    # Left teeth: mesial points right.  Right teeth: mesial points left.
    desired_mesial_x = torch.where(tooth_x < mid_x, torch.ones_like(tooth_x), -torch.ones_like(tooth_x))
    raw_u_axis_x = torch.cos(rboxes[:, 4])
    flip = (raw_u_axis_x * desired_mesial_x) < 0
    return flip & (~near_midline)


def _apply_u_flip(u: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
    if flip is None:
        return u
    return torch.where(flip, 1.0 - u, u)


def _u_axes_from_rboxes(rboxes: torch.Tensor, u_flip: Optional[torch.Tensor] = None,
                        image_aspect: float = 1.0) -> torch.Tensor:
    """Return normalized-image local-u unit vectors for each rbox.

    Rbox angles are defined in the aspect-corrected frame where x is multiplied by
    W/H.  To move sampling points in the original normalized image frame, convert
    (cos(angle), sin(angle)) back to (cos/a, sin) and renormalize.  u_flip makes
    the positive direction consistent with the configured mesial side.
    """
    if rboxes.numel() == 0:
        return rboxes.new_zeros((0, 2))
    a = float(image_aspect) if float(image_aspect) > 0 else 1.0
    angle = rboxes[:, 4]
    ux = torch.cos(angle) / a
    uy = torch.sin(angle)
    axes = torch.stack([ux, uy], dim=-1)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    if u_flip is not None:
        flip = u_flip.to(device=axes.device, dtype=torch.bool).reshape(-1)
        if flip.numel() == axes.shape[0]:
            axes = torch.where(flip[:, None], -axes, axes)
    return axes


def _corners_to_rbox_uv(corners_xy: torch.Tensor, rbox: torch.Tensor, u_flip: bool = False,
                        image_aspect: float = 1.0) -> torch.Tensor:
    a = float(image_aspect) if float(image_aspect) > 0 else 1.0
    cx, cy, rw, rh, angle = rbox
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    dx = (corners_xy[:, 0] - cx) * a
    dy = corners_xy[:, 1] - cy
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    u = local_x / rw.clamp(min=1e-6) + 0.5
    v = local_y / rh.clamp(min=1e-6) + 0.5
    if u_flip:
        u = 1.0 - u
    return torch.stack([u, v], dim=-1)



class StatisticalToothRiskMap(nn.Module):
    """Training-set statistical tooth risk + epoch-level hard-miss feedback, made semi-learnable.

    The statistical p / q maps stay as *frozen* buffers (the interpretable anchor estimated from the
    training set and refreshed only by ``compute_pq_statistics`` / hard-miss EMA).  On top of them we
    add zero-initialized learnable residuals so the network can fine-tune the prior where its own
    experience disagrees with the raw statistics, without ever drifting far from the anchor (an L2
    penalty pulls the residuals back to 0).  At initialization the residuals are exactly 0, so the
    effective maps equal the pure statistics and behaviour is identical to the non-learnable version.
    """

    def __init__(self, num_types: int = 6, size: int = 16,
                 hard_miss_confidence_k: float = 10.0, hard_miss_ema_momentum: float = 0.9):
        super().__init__()
        self.num_types = int(num_types)
        self.size = int(size)

        self.register_buffer("p_stat", torch.full((self.num_types,), 1e-4, dtype=torch.float32))
        self.register_buffer("q_stat", torch.full((self.num_types, 1, self.size, self.size), 1.0 / (self.size * self.size), dtype=torch.float32))
        self.register_buffer("h_stat", torch.zeros((self.num_types, 1, self.size, self.size), dtype=torch.float32))
        self.register_buffer("risk_stats_initialized", torch.zeros((), dtype=torch.bool))

        # Semi-learnable residuals (zero-init): q_eff = softmax(log q_stat + q_residual),
        # p_eff = sigmoid(logit p_stat + p_residual).  Regularized toward 0 by residual_reg_loss().
        self.q_residual = nn.Parameter(torch.zeros((self.num_types, 1, self.size, self.size), dtype=torch.float32))
        self.p_residual = nn.Parameter(torch.zeros((self.num_types,), dtype=torch.float32))

        # Hard-miss hyper-parameters are configurable (surfaced in the runtime config).
        self.hard_miss_confidence_k = float(hard_miss_confidence_k)
        self.hard_miss_ema_momentum = float(hard_miss_ema_momentum)

    def effective_maps(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Frozen statistical anchor (no grad) + zero-init learnable residual.
        p_base = self.p_stat.detach().clamp(1e-4, 1.0 - 1e-4)
        p_logit = torch.log(p_base) - torch.log1p(-p_base)
        p_eff = torch.sigmoid(p_logit + self.p_residual).clamp(1e-4, 1.0 - 1e-4)

        q_base = self.q_stat.detach().clamp_min(1e-6)
        q_log = torch.log(q_base)
        q_eff = F.softmax((q_log + self.q_residual).flatten(1), dim=1).view_as(q_base)
        r_eff = p_eff.view(-1, 1, 1, 1) * q_eff
        return p_eff, q_eff, r_eff

    def residual_reg_loss(self) -> torch.Tensor:
        """Unweighted L2 anchor penalty keeping the learnable residuals near the statistical base."""
        return self.q_residual.pow(2).mean() + self.p_residual.pow(2).mean()

    @staticmethod
    def normalize_per_type(map_tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        flat = map_tensor.flatten(1)
        mn = flat.min(dim=1).values.view(-1, 1, 1, 1)
        mx = flat.max(dim=1).values.view(-1, 1, 1, 1)
        denom = mx - mn
        norm = (map_tensor - mn) / (denom + eps)
        return torch.where(denom > eps, norm, torch.zeros_like(norm))

def _normalized_center_distance(points_xy: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Distance from points [N,2] to box centers [M,4], normalized by box diagonal. Returns [M,N]."""
    if boxes_xyxy.numel() == 0 or points_xy.numel() == 0:
        return boxes_xyxy.new_zeros((boxes_xyxy.shape[0], points_xy.shape[0]))
    centers = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) * 0.5
    wh = (boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]).clamp(min=1e-6)
    diag = torch.sqrt((wh * wh).sum(dim=-1)).clamp(min=1e-6)
    d = torch.cdist(centers, points_xy.to(boxes_xyxy.dtype))
    return d / diag[:, None]


def _assignment_score(
    points_xy: torch.Tensor,
    gt_xyxy: Optional[torch.Tensor],
    tooth_xyxy: torch.Tensor,
    tooth_rboxes: Optional[torch.Tensor] = None,
    tooth_scores: Optional[torch.Tensor] = None,
    rotated_inside_bonus_weight: float = 2.0,
    iou_weight: float = 1.0,
    dist_weight: float = 1.0,
    conf_weight: float = 0.5,
    image_aspect: float = 1.0,
) -> torch.Tensor:
    """Score candidate tooth boxes for a point/GT. Returns [M, N] when N points are provided."""
    m = tooth_xyxy.shape[0]
    n = points_xy.shape[0]
    if m == 0 or n == 0:
        return tooth_xyxy.new_zeros((m, n))
    dist = _normalized_center_distance(points_xy, tooth_xyxy)
    score = -float(dist_weight) * dist
    if tooth_scores is not None and tooth_scores.numel() == m:
        score = score + float(conf_weight) * tooth_scores.reshape(-1, 1).to(score.dtype)
    if tooth_rboxes is not None and tooth_rboxes.numel() > 0:
        inside_r, _, _ = _rotated_inside(points_xy[:, 0], points_xy[:, 1], tooth_rboxes, image_aspect=image_aspect)
        score = score + float(rotated_inside_bonus_weight) * inside_r.to(score.dtype)
    if gt_xyxy is not None and gt_xyxy.numel() == 4:
        iou = torchvision.ops.box_iou(gt_xyxy.view(1, 4), tooth_xyxy).view(-1)
        score = score + float(iou_weight) * iou.reshape(-1, 1).to(score.dtype)
    return score


def _expanded_xyxy(boxes_xyxy: torch.Tensor, ratio: float) -> torch.Tensor:
    return expand_xyxy_boxes(boxes_xyxy, float(ratio)).clamp(0, 1)


def _assign_gt_to_tooth(
    gt_xyxy: torch.Tensor,
    tooth_xyxy: torch.Tensor,
    tooth_rboxes: Optional[torch.Tensor] = None,
    tooth_scores: Optional[torch.Tensor] = None,
    expand_ratio: float = 1.1,
    image_aspect: float = 1.0,
) -> int:
    if tooth_xyxy.numel() == 0:
        return -1
    center = (gt_xyxy[:2] + gt_xyxy[2:]) * 0.5
    points = center.view(1, 2)
    x, y = center[0], center[1]

    score = _assignment_score(points, gt_xyxy, tooth_xyxy, tooth_rboxes, tooth_scores, image_aspect=image_aspect)

    if tooth_rboxes is not None and tooth_rboxes.numel() > 0:
        inside_r, _, _ = _rotated_inside(x.view(1), y.view(1), tooth_rboxes, image_aspect=image_aspect)
        mask = inside_r[:, 0]
        if mask.any():
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            return int(idx[score[idx, 0].argmax()].item())

    inside = (x >= tooth_xyxy[:, 0]) & (x <= tooth_xyxy[:, 2]) & (y >= tooth_xyxy[:, 1]) & (y <= tooth_xyxy[:, 3])
    if inside.any():
        idx = torch.nonzero(inside, as_tuple=False).flatten()
        return int(idx[score[idx, 0].argmax()].item())

    expanded = _expanded_xyxy(tooth_xyxy, expand_ratio)
    inside_exp = (x >= expanded[:, 0]) & (x <= expanded[:, 2]) & (y >= expanded[:, 1]) & (y <= expanded[:, 3])
    if inside_exp.any():
        idx = torch.nonzero(inside_exp, as_tuple=False).flatten()
        return int(idx[score[idx, 0].argmax()].item())

    ious = torchvision.ops.box_iou(gt_xyxy.view(1, 4), tooth_xyxy).view(-1)
    if ious.numel() > 0 and float(ious.max()) > 0:
        return int(ious.argmax().item())

    dist = _normalized_center_distance(points, tooth_xyxy)[:, 0]
    return int(dist.argmin().item()) if dist.numel() > 0 else -1


def _rboxes_for_loss(target: dict, tooth_xyxy: torch.Tensor, tooth_types: torch.Tensor, device, dtype,
                     image_aspect: float = 1.0) -> torch.Tensor:
    if "tooth_rboxes" in target:
        rboxes = target["tooth_rboxes"].to(device=device, dtype=dtype).reshape(-1, 5)
        if rboxes.shape[0] == tooth_xyxy.shape[0]:
            return rboxes

    if "tooth_masks" not in target:
        return _axis_aligned_rboxes_from_xyxy(tooth_xyxy, tooth_types, image_aspect=image_aspect).to(device=device, dtype=dtype)

    rboxes, _ = _rboxes_from_masks(target.get("tooth_masks"), tooth_xyxy, tooth_types=tooth_types, image_aspect=image_aspect)
    return rboxes.to(device=device, dtype=dtype)


def _gaussian_soft_target(
    center_u: torch.Tensor,
    center_v: torch.Tensor,
    w_local: torch.Tensor,
    h_local: torch.Tensor,
    size: int,
    sigma_min: float = 1.0,
    sigma_scale: float = 0.5,
    normalize: str = "sum",
) -> torch.Tensor:
    device = center_u.device
    dtype = center_u.dtype
    s = int(size)
    yy, xx = torch.meshgrid(torch.arange(s, device=device, dtype=dtype), torch.arange(s, device=device, dtype=dtype), indexing="ij")
    cu = center_u.clamp(0, 1) * (s - 1)
    cv = center_v.clamp(0, 1) * (s - 1)
    sig_u = torch.clamp(w_local * float(sigma_scale), min=float(sigma_min))
    sig_v = torch.clamp(h_local * float(sigma_scale), min=float(sigma_min))
    g = torch.exp(-0.5 * (((xx - cu) / sig_u.clamp(min=1e-6)) ** 2 + ((yy - cv) / sig_v.clamp(min=1e-6)) ** 2))
    if normalize == "peak":
        return g / g.max().clamp_min(1e-6)
    return g / (g.sum() + 1e-6)


def _gt_local_gaussian(
    gt_xyxy: torch.Tensor,
    rbox: torch.Tensor,
    u_flip: bool,
    size: int,
    sigma_min: float,
    sigma_scale: float,
    normalize: str = "sum",
    image_aspect: float = 1.0,
) -> torch.Tensor:
    x1, y1, x2, y2 = gt_xyxy
    corners = torch.stack([
        torch.stack([x1, y1]),
        torch.stack([x2, y1]),
        torch.stack([x2, y2]),
        torch.stack([x1, y2]),
    ], dim=0)
    uv = _corners_to_rbox_uv(corners, rbox, u_flip, image_aspect=image_aspect).clamp(0, 1)
    center = (gt_xyxy[:2] + gt_xyxy[2:]) * 0.5
    center_uv = _corners_to_rbox_uv(center.view(1, 2), rbox, u_flip, image_aspect=image_aspect).view(-1).clamp(0, 1)
    w_local = ((uv[:, 0].max() - uv[:, 0].min()).clamp(min=1.0 / size) * size)
    h_local = ((uv[:, 1].max() - uv[:, 1].min()).clamp(min=1.0 / size) * size)
    return _gaussian_soft_target(center_uv[0], center_uv[1], w_local, h_local, size, sigma_min, sigma_scale, normalize=normalize)


def _sample_single_heatmap(heatmap: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if u.numel() == 0:
        return u.new_zeros((0,))
    grid = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1).view(1, -1, 1, 2)
    return F.grid_sample(heatmap.view(1, 1, heatmap.shape[-2], heatmap.shape[-1]), grid,
                         mode="bilinear", align_corners=True).view(-1)


@torch.no_grad()
def gt_hard_weights_for_target(
    target: dict,
    h_stat: torch.Tensor,
    image_aspect: float = 1.0,
    beta: float = 1.0,
    expand_ratio: float = 1.1,
    num_types: int = 6,
) -> torch.Tensor:
    """Per-GT positive-loss multiplier ``1 + beta * h(u, v | t)`` for hard-miss modulation (#6).

    ``h`` is the epoch-level hard-miss map, min-max normalized per tooth type to [0, 1] so the
    multiplier is scale-robust and lies in ``[1, 1 + beta]``.  Caries GTs that fall on tooth-local
    regions the model has systematically missed get a larger weight, focusing optimization there.
    Returns ones (no effect) when there is no hard-miss signal yet, no tooth prompt, or ``beta <= 0``
    -- so the modulation is a strict no-op until the statistics are meaningful.
    """
    boxes = target.get("boxes")
    device = h_stat.device
    n_gt = 0 if boxes is None else int(boxes.reshape(-1, 4).shape[0])
    ones = torch.ones((n_gt,), device=device, dtype=torch.float32)
    if n_gt == 0 or float(beta) <= 0.0:
        return ones
    if "tooth_boxes" not in target or target["tooth_boxes"].numel() == 0:
        return ones
    if not torch.isfinite(h_stat).all() or float(h_stat.abs().sum().item()) <= 0.0:
        return ones

    tooth_boxes = target["tooth_boxes"].to(device=device, dtype=torch.float32).reshape(-1, 4)
    tooth_xyxy = cxcywh_to_xyxy(tooth_boxes).clamp(0, 1)
    tooth_types = target.get("tooth_types", torch.zeros((tooth_xyxy.shape[0],), device=device, dtype=torch.long)
                             ).to(device=device).reshape(-1).long().clamp(0, num_types - 1)
    tooth_scores = target.get("tooth_scores", torch.ones((tooth_xyxy.shape[0],), device=device, dtype=torch.float32)
                              ).to(device=device, dtype=torch.float32).reshape(-1)
    rboxes = _rboxes_for_loss(target, tooth_xyxy, tooth_types, device, torch.float32, image_aspect=image_aspect)
    midline_boxes = None
    if "tooth_midline_boxes" in target:
        mb = target["tooth_midline_boxes"].to(device=device, dtype=torch.float32).reshape(-1, 4)
        if mb.shape[0] == tooth_xyxy.shape[0]:
            midline_boxes = mb
    u_flip = _mesial_u_flip(rboxes, midline_boxes)

    h_norm = StatisticalToothRiskMap.normalize_per_type(h_stat.to(device=device, dtype=torch.float32))
    gt_xyxy = cxcywh_to_xyxy(boxes.to(device=device, dtype=torch.float32).reshape(-1, 4)).clamp(0, 1)
    weights = ones.clone()
    for j, gt in enumerate(gt_xyxy):
        ti = _assign_gt_to_tooth(gt, tooth_xyxy, tooth_rboxes=rboxes, tooth_scores=tooth_scores,
                                 expand_ratio=expand_ratio, image_aspect=image_aspect)
        if ti < 0:
            continue
        center = ((gt[:2] + gt[2:]) * 0.5).view(1, 2)
        uv = _corners_to_rbox_uv(center, rboxes[ti], bool(u_flip[ti].item()), image_aspect=image_aspect).view(-1).clamp(0, 1)
        tid = int(tooth_types[ti].item())
        hv = _sample_single_heatmap(h_norm[tid, 0], uv[0:1], uv[1:2]).clamp(0, 1).reshape(())
        weights[j] = 1.0 + float(beta) * hv
    return weights


class ToothPriorBuilder(nn.Module):
    def __init__(
        self,
        template_size: int = 16,
        context_expand_ratio: float = 1.1,
        context_side_expand: float = 0.05,
        context_min_cell: int = 1,
        num_tooth_types: int = 6,
        use_tooth_rotated_box: bool = True,
        tooth_rbox_quantile_low: float = 0.05,
        tooth_rbox_quantile_high: float = 0.95,
        tooth_rbox_expand_ratio: float = 1.05,
        tooth_rbox_min_points: int = 12,
        tooth_rbox_max_points: int = 512,
        tooth_rbox_bbox_clip_margin: float = 0.02,
        tooth_expanded_box_ratio: Optional[float] = None,
        risk_gaussian_sigma_min: float = 1.0,
        risk_gaussian_sigma_scale: float = 0.5,
        hard_miss_confidence_k: float = 10.0,
        hard_miss_ema_momentum: float = 0.9,
        upper_tooth_types=None,
        lower_tooth_types=None,
    ):
        super().__init__()
        self.template_size = int(template_size)
        self.tooth_expanded_box_ratio = float(tooth_expanded_box_ratio if tooth_expanded_box_ratio is not None else context_expand_ratio)
        self.context_expand_ratio = self.tooth_expanded_box_ratio
        self.context_side_expand = float(context_side_expand)
        self.context_min_cell = int(context_min_cell)
        self.num_tooth_types = int(num_tooth_types)
        self.upper_tooth_types = _normalize_type_tuple(upper_tooth_types, DEFAULT_UPPER_TOOTH_TYPES)
        self.lower_tooth_types = _normalize_type_tuple(lower_tooth_types, DEFAULT_LOWER_TOOTH_TYPES)
        self.use_tooth_rotated_box = bool(use_tooth_rotated_box)
        self.tooth_rbox_quantile_low = float(tooth_rbox_quantile_low)
        self.tooth_rbox_quantile_high = float(tooth_rbox_quantile_high)
        self.tooth_rbox_expand_ratio = float(tooth_rbox_expand_ratio)
        self.tooth_rbox_min_points = int(tooth_rbox_min_points)
        self.tooth_rbox_max_points = int(tooth_rbox_max_points)
        self.tooth_rbox_bbox_clip_margin = float(tooth_rbox_bbox_clip_margin)
        self.risk_gaussian_sigma_min = float(risk_gaussian_sigma_min)
        self.risk_gaussian_sigma_scale = float(risk_gaussian_sigma_scale)
        self.risk_map = StatisticalToothRiskMap(
            num_types=self.num_tooth_types,
            size=self.template_size,
            hard_miss_confidence_k=float(hard_miss_confidence_k),
            hard_miss_ema_momentum=float(hard_miss_ema_momentum),
        )

    @property
    def p_stat(self):
        return self.risk_map.p_stat

    @property
    def q_stat(self):
        return self.risk_map.q_stat

    @property
    def h_stat(self):
        return self.risk_map.h_stat

    def effective_maps(self):
        return self.risk_map.effective_maps()

    @staticmethod
    def _empty_priors(batch_size: int, length: int, device, dtype) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros((batch_size, length), device=device, dtype=dtype)
        minus = torch.full((batch_size, length), -1, device=device, dtype=torch.long)
        return {
            "risk_prior": zeros,
            "risk_prior_norm": zeros.clone(),
            "risk_score_prior": zeros.clone(),
            "hard_prior": zeros.clone(),
            "image_risk_prior": zeros.clone(),
            "tooth_prior": zeros.clone(),
            "context_prior": zeros.clone(),
            "point_u": zeros.clone(),
            "point_v": zeros.clone(),
            "point_u_axis_x": zeros.clone(),
            "point_u_axis_y": zeros.clone(),
            "point_midline_x": zeros.clone(),
            "risk_template_target": zeros.clone(),
            "risk_template_mask": torch.zeros((batch_size, length), device=device, dtype=torch.bool),
            "tooth_inside_mask": torch.zeros((batch_size, length), device=device, dtype=torch.bool),
            "context_mask": torch.zeros((batch_size, length), device=device, dtype=torch.bool),
            "global_mask": torch.ones((batch_size, length), device=device, dtype=torch.bool),
            "point_type": minus.clone(),
            "point_tooth": minus.clone(),
        }

    def _get_grids(self, spatial_shapes, device, dtype):
        key = tuple((int(h), int(w)) for h, w in spatial_shapes)
        cache = getattr(self, "_grid_cache", None)
        if cache is not None and cache["key"] == key \
           and cache["x"][0].device == device and cache["x"][0].dtype == dtype:
            return cache["x"], cache["y"]
        xs, ys = [], []
        for (h, w) in spatial_shapes:
            h, w = int(h), int(w)
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype), indexing="ij")
            xs.append(((xx + 0.5) / w).reshape(-1))
            ys.append(((yy + 0.5) / h).reshape(-1))
        self._grid_cache = {"key": key, "x": xs, "y": ys}
        return xs, ys

    def _rboxes_from_target(self, target: dict, boxes_xyxy: torch.Tensor, tooth_types: torch.Tensor, device, dtype,
                           image_aspect: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        fallback = _axis_aligned_rboxes_from_xyxy(
            boxes_xyxy, tooth_types, image_aspect=image_aspect,
            upper_tooth_types=self.upper_tooth_types, lower_tooth_types=self.lower_tooth_types
        ).to(device=device, dtype=dtype)
        invalid = torch.zeros((boxes_xyxy.shape[0],), device=device, dtype=torch.bool)

        if not self.use_tooth_rotated_box:
            return fallback, invalid

        if "tooth_rboxes" in target:
            rboxes = target["tooth_rboxes"].to(device=device, dtype=dtype).reshape(-1, 5)
            if rboxes.shape[0] == boxes_xyxy.shape[0]:
                valid = target.get("tooth_rbox_valid", None)
                if valid is not None:
                    valid = valid.to(device=device, dtype=torch.bool).reshape(-1)
                    if valid.numel() == boxes_xyxy.shape[0]:
                        return rboxes, valid
                # If an external path provides rboxes without validity flags, treat them as real
                # rotated boxes instead of silently mixing them with axis-aligned fallback.
                return rboxes, torch.ones((boxes_xyxy.shape[0],), device=device, dtype=torch.bool)

        if "tooth_masks" not in target:
            return fallback, invalid

        rboxes, valid = _rboxes_from_masks(
            target.get("tooth_masks"),
            boxes_xyxy,
            tooth_types=tooth_types,
            quantile_low=self.tooth_rbox_quantile_low,
            quantile_high=self.tooth_rbox_quantile_high,
            expand_ratio=self.tooth_rbox_expand_ratio,
            min_points=self.tooth_rbox_min_points,
            max_points=self.tooth_rbox_max_points,
            bbox_clip_margin=self.tooth_rbox_bbox_clip_margin,
            image_aspect=image_aspect,
            upper_tooth_types=self.upper_tooth_types,
            lower_tooth_types=self.lower_tooth_types,
        )
        return rboxes.to(device=device, dtype=dtype), valid.to(device=device, dtype=torch.bool)

    def _prepare_targets(self, targets, device, dtype, image_aspect: float = 1.0):
        prepared = []
        num_types = self.num_tooth_types
        for target in targets:
            if "tooth_boxes" not in target:
                prepared.append(None)
                continue
            boxes = target["tooth_boxes"].to(device=device, dtype=dtype).reshape(-1, 4)
            if boxes.numel() == 0:
                prepared.append(None)
                continue
            # Prefer the aspect stored alongside the precomputed rbox; fall back to the caller's value.
            a = _aspect_from_target(target, default=float(image_aspect))
            scores = target.get("tooth_scores", torch.ones((boxes.shape[0],), device=device, dtype=dtype)
                                ).to(device=device, dtype=dtype).reshape(-1).clamp(0, 1)
            types = target.get("tooth_types", torch.zeros((boxes.shape[0],), device=device, dtype=torch.long)
                               ).to(device=device).reshape(-1).long().clamp(0, num_types - 1)
            boxes_xyxy = cxcywh_to_xyxy(boxes).clamp(0, 1)
            rboxes, rbox_valid = self._rboxes_from_target(target, boxes_xyxy, types, device, dtype, image_aspect=a)
            midline_boxes = None
            if "tooth_midline_boxes" in target:
                midline_boxes = target["tooth_midline_boxes"].to(device=device, dtype=dtype).reshape(-1, 4)
                if midline_boxes.shape[0] != boxes.shape[0]:
                    midline_boxes = None
            u_flip = _mesial_u_flip(rboxes, midline_boxes)
            # #4 anatomical midline x (normalized) in the current network frame, tracked through all
            # geometric augmentations via the tiny midline box.  Used to mirror a query to its
            # contralateral same-type tooth as a negative reference.  Falls back to image center.
            if midline_boxes is not None and midline_boxes.numel() > 0:
                midline_x = float(((midline_boxes[:, 0] + midline_boxes[:, 2]) * 0.5).mean().item())
            else:
                midline_x = 0.5
            prepared.append(dict(
                boxes_xyxy=boxes_xyxy, rboxes=rboxes, rbox_valid=rbox_valid, scores=scores, types=types,
                midline_boxes=midline_boxes, u_flip=u_flip, aspect=float(a), midline_x=float(midline_x)))
        return prepared

    def _sample_map(self, maps: torch.Tensor, type_ids: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if type_ids.numel() == 0:
            return u.new_zeros((0,))
        type_ids = type_ids.clamp(0, maps.shape[0] - 1).long()
        grid = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1).view(-1, 1, 1, 2)
        return F.grid_sample(maps[type_ids], grid, mode="bilinear", align_corners=True).view(-1)

    def forward(self, targets, spatial_shapes, device, dtype):
        batch_size = len(targets) if targets is not None else 0
        length = sum(int(h) * int(w) for h, w in spatial_shapes)
        p_eff, q_eff, r_eff = self.risk_map.effective_maps()
        r_norm = self.risk_map.normalize_per_type(r_eff)
        h_norm = self.risk_map.normalize_per_type(self.risk_map.h_stat.to(device=r_eff.device, dtype=r_eff.dtype))

        zeros = torch.zeros((batch_size, length), device=device, dtype=dtype)
        priors = {
            "risk_prior": zeros.clone(),
            "risk_prior_norm": zeros.clone(),
            "risk_score_prior": zeros.clone(),
            "hard_prior": zeros.clone(),
            "image_risk_prior": zeros.clone(),
            "tooth_prior": zeros.clone(),
            "context_prior": zeros.clone(),
            "point_u": zeros.clone(),
            "point_v": zeros.clone(),
            "point_u_axis_x": zeros.clone(),
            "point_u_axis_y": zeros.clone(),
            "point_midline_x": zeros.clone(),
            "risk_template_target": zeros.clone(),
            "risk_template_mask": torch.zeros((batch_size, length), device=device, dtype=torch.bool),
            "tooth_inside_mask": torch.zeros((batch_size, length), device=device, dtype=torch.bool),
            "context_mask": torch.zeros((batch_size, length), device=device, dtype=torch.bool),
            "global_mask": torch.ones((batch_size, length), device=device, dtype=torch.bool),
            "point_type": torch.full((batch_size, length), -1, device=device, dtype=torch.long),
            "point_tooth": torch.full((batch_size, length), -1, device=device, dtype=torch.long),
            "p_eff": p_eff,
            "r_eff": r_eff,
        }
        if targets is None or batch_size == 0:
            return priors

        xs, ys = self._get_grids(spatial_shapes, device, dtype)
        image_aspect = _aspect_from_spatial_shapes(spatial_shapes)
        prepared_targets = self._prepare_targets(targets, device, dtype, image_aspect=image_aspect)
        r_eff = r_eff.to(device=device, dtype=dtype)
        r_norm = r_norm.to(device=device, dtype=dtype)
        h_norm = h_norm.to(device=device, dtype=dtype)

        offset = 0
        for level_idx, (h, w) in enumerate(spatial_shapes):
            h, w = int(h), int(w)
            level_len = h * w
            x, y = xs[level_idx], ys[level_idx]
            points = torch.stack([x, y], dim=-1)
            for b, prepared in enumerate(prepared_targets):
                if prepared is None:
                    continue
                rboxes = prepared["rboxes"]
                rbox_valid = prepared.get("rbox_valid", torch.zeros((rboxes.shape[0],), device=device, dtype=torch.bool))
                boxes_xyxy = prepared["boxes_xyxy"]
                scores = prepared["scores"]
                types = prepared["types"]
                u_flip = prepared["u_flip"]
                u_axes = _u_axes_from_rboxes(rboxes, u_flip, image_aspect=image_aspect)
                num_teeth = boxes_xyxy.shape[0]
                if num_teeth == 0:
                    continue

                inside_r, local_x, local_y = _rotated_inside(x, y, rboxes, image_aspect=image_aspect)
                # Axis-aligned fallback only where no rotated assignment exists.
                x_row, y_row = x[None, :], y[None, :]
                inside_a = (x_row >= boxes_xyxy[:, 0:1]) & (x_row <= boxes_xyxy[:, 2:3]) & \
                           (y_row >= boxes_xyxy[:, 1:2]) & (y_row <= boxes_xyxy[:, 3:4])
                # Use the mask-PCA rotated box as the hard tooth-inside gate whenever it is valid.
                # Axis-aligned boxes are only a fallback for teeth whose masks could not produce a
                # reliable rbox; otherwise tilted teeth would again include large background regions.
                inside = torch.where(rbox_valid[:, None], inside_r, inside_a)
                any_inside = inside.any(dim=0)
                flat_all = offset + torch.arange(level_len, device=device)

                if any_inside.any():
                    candidate_points = points[any_inside]
                    cand_idx = torch.nonzero(any_inside, as_tuple=False).flatten()
                    assign_scores = _assignment_score(
                        candidate_points, None, boxes_xyxy, rboxes, scores,
                        rotated_inside_bonus_weight=1.0, iou_weight=0.0, dist_weight=1.0, conf_weight=1.0,
                        image_aspect=image_aspect)
                    assign_scores = assign_scores.masked_fill(~inside[:, cand_idx], -torch.inf)
                    best_tooth = assign_scores.argmax(dim=0)
                    chosen_valid = torch.isfinite(assign_scores.gather(0, best_tooth.view(1, -1)).view(-1))
                    if chosen_valid.any():
                        cand_idx = cand_idx[chosen_valid]
                        best_tooth = best_tooth[chosen_valid]
                        flat = offset + cand_idx
                        rw = rboxes[:, 2].clamp(min=1e-6)
                        rh = rboxes[:, 3].clamp(min=1e-6)
                        lx = local_x[best_tooth, cand_idx]
                        ly = local_y[best_tooth, cand_idx]
                        # For points that only matched axis-aligned fallback, compute normalized axis-aligned local coords.
                        chosen_inside_r = inside_r[best_tooth, cand_idx]
                        ax_u = ((x[cand_idx] - boxes_xyxy[best_tooth, 0]) /
                                (boxes_xyxy[best_tooth, 2] - boxes_xyxy[best_tooth, 0]).clamp(min=1e-6)).clamp(0, 1)
                        ax_v = ((y[cand_idx] - boxes_xyxy[best_tooth, 1]) /
                                (boxes_xyxy[best_tooth, 3] - boxes_xyxy[best_tooth, 1]).clamp(min=1e-6)).clamp(0, 1)
                        rot_u = (lx / rw[best_tooth] + 0.5).clamp(0, 1)
                        rot_u = _apply_u_flip(rot_u, u_flip[best_tooth])
                        rot_v = (ly / rh[best_tooth] + 0.5).clamp(0, 1)
                        u = torch.where(chosen_inside_r, rot_u, ax_u)
                        v = torch.where(chosen_inside_r, rot_v, ax_v)
                        point_types = types[best_tooth]
                        sampled_r = self._sample_map(r_eff, point_types, u, v)
                        sampled_rn = self._sample_map(r_norm, point_types, u, v)
                        sampled_hn = self._sample_map(h_norm, point_types, u, v)

                        priors["risk_prior"][b, flat] = sampled_r
                        priors["risk_prior_norm"][b, flat] = sampled_rn
                        priors["hard_prior"][b, flat] = sampled_hn
                        priors["tooth_prior"][b, flat] = 1.0
                        priors["tooth_inside_mask"][b, flat] = True
                        priors["risk_template_mask"][b, flat] = True
                        priors["point_type"][b, flat] = point_types
                        priors["point_tooth"][b, flat] = best_tooth.long()
                        priors["point_u"][b, flat] = u
                        priors["point_v"][b, flat] = v
                        priors["point_u_axis_x"][b, flat] = u_axes[best_tooth, 0]
                        priors["point_u_axis_y"][b, flat] = u_axes[best_tooth, 1]
                        priors["point_midline_x"][b, flat] = float(prepared.get("midline_x", 0.5))

                # Context query region: rotated local left/right strips only.
                # u in [-side, 0) U (1, 1+side], v in [0, 1].
                side = max(float(self.context_side_expand), 0.0)
                if side > 0 and rboxes.numel() > 0:
                    rw = rboxes[:, 2].clamp(min=1e-6)[:, None]
                    rh = rboxes[:, 3].clamp(min=1e-6)[:, None]
                    ctx_u_raw = local_x / rw + 0.5
                    ctx_u = _apply_u_flip(ctx_u_raw, u_flip[:, None])
                    ctx_v = local_y / rh + 0.5
                    side_ctx = (((ctx_u >= -side) & (ctx_u < 0.0)) | ((ctx_u > 1.0) & (ctx_u <= 1.0 + side))) & \
                               (ctx_v >= 0.0) & (ctx_v <= 1.0)
                    side_ctx = side_ctx & (~inside_r)
                    ctx_union = side_ctx.any(dim=0)
                    if ctx_union.any():
                        ctx_idx = torch.nonzero(ctx_union, as_tuple=False).flatten()
                        ctx_scores = _assignment_score(points[ctx_idx], None, boxes_xyxy, rboxes, scores,
                                                       rotated_inside_bonus_weight=0.0, iou_weight=0.0,
                                                       dist_weight=1.0, conf_weight=1.0,
                                                       image_aspect=image_aspect)
                        ctx_scores = ctx_scores.masked_fill(~side_ctx[:, ctx_idx], -torch.inf)
                        ctx_best = ctx_scores.argmax(dim=0)
                        ctx_valid = torch.isfinite(ctx_scores.gather(0, ctx_best.view(1, -1)).view(-1))
                        if ctx_valid.any():
                            ctx_idx = ctx_idx[ctx_valid]
                            ctx_best = ctx_best[ctx_valid]
                            flat_ctx = offset + ctx_idx
                            cu = ctx_u[ctx_best, ctx_idx]
                            cv = ctx_v[ctx_best, ctx_idx].clamp(0, 1)
                            priors["context_mask"][b, flat_ctx] = True
                            priors["context_prior"][b, flat_ctx] = 1.0
                            priors["point_type"][b, flat_ctx] = types[ctx_best]
                            priors["point_tooth"][b, flat_ctx] = ctx_best.long()
                            priors["point_u"][b, flat_ctx] = cu
                            priors["point_v"][b, flat_ctx] = cv
                            priors["point_u_axis_x"][b, flat_ctx] = u_axes[ctx_best, 0]
                            priors["point_u_axis_y"][b, flat_ctx] = u_axes[ctx_best, 1]
                            priors["point_midline_x"][b, flat_ctx] = float(prepared.get("midline_x", 0.5))

                expanded = _expanded_xyxy(boxes_xyxy, self.tooth_expanded_box_ratio)
                inside_exp = (x_row >= expanded[:, 0:1]) & (x_row <= expanded[:, 2:3]) & \
                             (y_row >= expanded[:, 1:2]) & (y_row <= expanded[:, 3:4])
                expanded_union = inside_exp.any(dim=0)
                priors["global_mask"][b, offset:offset + level_len] &= ~expanded_union
            offset += level_len

        self._fill_risk_template_targets(targets, prepared_targets, priors, device, dtype)
        return priors

    def _fill_risk_template_targets(self, targets, prepared_targets, priors: Dict[str, torch.Tensor], device, dtype):
        if targets is None or prepared_targets is None:
            return
        s = self.template_size
        for b, (target, prep) in enumerate(zip(targets, prepared_targets)):
            if prep is None:
                continue
            gt_boxes = target.get("boxes")
            if gt_boxes is None or gt_boxes.numel() == 0:
                continue
            tooth_xyxy = prep["boxes_xyxy"]
            tooth_types = prep["types"]
            tooth_rboxes = prep["rboxes"]
            tooth_scores = prep["scores"]
            u_flip = prep["u_flip"]
            a = float(prep.get("aspect", 1.0))
            gt_xyxy = cxcywh_to_xyxy(gt_boxes.to(device=device, dtype=dtype).reshape(-1, 4)).clamp(0, 1)
            for gt in gt_xyxy:
                tooth_idx = _assign_gt_to_tooth(gt, tooth_xyxy, tooth_rboxes=tooth_rboxes, tooth_scores=tooth_scores,
                                                expand_ratio=self.tooth_expanded_box_ratio, image_aspect=a)
                if tooth_idx < 0:
                    continue
                heatmap = _gt_local_gaussian(
                    gt, tooth_rboxes[tooth_idx], bool(u_flip[tooth_idx].item()), s,
                    self.risk_gaussian_sigma_min, self.risk_gaussian_sigma_scale, normalize="peak", image_aspect=a)
                point_mask = (priors["point_tooth"][b] == int(tooth_idx)) & priors["tooth_inside_mask"][b]
                if not point_mask.any():
                    continue
                u = priors["point_u"][b, point_mask].clamp(0, 1)
                v = priors["point_v"][b, point_mask].clamp(0, 1)
                values = _sample_single_heatmap(heatmap.to(device=device, dtype=dtype), u, v)
                priors["risk_template_target"][b, point_mask] = torch.maximum(
                    priors["risk_template_target"][b, point_mask], values.to(priors["risk_template_target"].dtype))

    @torch.no_grad()
    def compute_pq_statistics(self, data_loader, device):
        num_types = self.num_tooth_types
        s = self.template_size
        dtype = self.p_stat.dtype
        n_t = torch.zeros((num_types,), device=device, dtype=torch.float64)
        c_t = torch.zeros((num_types,), device=device, dtype=torch.float64)
        q_count = torch.zeros((num_types, 1, s, s), device=device, dtype=torch.float64)

        for _, targets in data_loader:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            prepared = self._prepare_targets(targets, device, torch.float32)
            for target, prep in zip(targets, prepared):
                if prep is None:
                    continue
                tooth_xyxy = prep["boxes_xyxy"]
                tooth_types = prep["types"]
                tooth_rboxes = prep["rboxes"]
                tooth_scores = prep["scores"]
                u_flip = prep["u_flip"]
                a = float(prep.get("aspect", 1.0))
                if tooth_xyxy.numel() == 0:
                    continue
                for t in tooth_types:
                    n_t[int(t.item())] += 1.0
                carious_tooth = torch.zeros((tooth_xyxy.shape[0],), device=device, dtype=torch.bool)
                gt_boxes = target.get("boxes")
                if gt_boxes is None or gt_boxes.numel() == 0:
                    continue
                gt_xyxy = cxcywh_to_xyxy(gt_boxes.to(device=device, dtype=torch.float32).reshape(-1, 4)).clamp(0, 1)
                for gt in gt_xyxy:
                    tooth_idx = _assign_gt_to_tooth(
                        gt, tooth_xyxy, tooth_rboxes=tooth_rboxes, tooth_scores=tooth_scores,
                        expand_ratio=self.tooth_expanded_box_ratio, image_aspect=a)
                    if tooth_idx < 0:
                        continue
                    carious_tooth[tooth_idx] = True
                    type_id = int(tooth_types[tooth_idx].item())
                    g = _gt_local_gaussian(
                        gt, tooth_rboxes[tooth_idx], bool(u_flip[tooth_idx].item()), s,
                        self.risk_gaussian_sigma_min, self.risk_gaussian_sigma_scale, image_aspect=a)
                    q_count[type_id, 0] += g.to(torch.float64)
                for idx in torch.nonzero(carious_tooth, as_tuple=False).flatten():
                    c_t[int(tooth_types[idx].item())] += 1.0

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(n_t)
            torch.distributed.all_reduce(c_t)
            torch.distributed.all_reduce(q_count)

        p = torch.where(n_t > 0, c_t / n_t.clamp(min=1.0), torch.full_like(n_t, 1e-4))
        q = torch.full_like(q_count, 1.0 / (s * s))
        q_sum = q_count.flatten(1).sum(dim=1)
        valid = q_sum > 0
        if valid.any():
            q[valid] = q_count[valid] / q_sum[valid].view(-1, 1, 1, 1)
        self.risk_map.p_stat.copy_(p.to(device=self.risk_map.p_stat.device, dtype=self.risk_map.p_stat.dtype).clamp(1e-4, 1 - 1e-4))
        self.risk_map.q_stat.copy_(q.to(device=self.risk_map.q_stat.device, dtype=self.risk_map.q_stat.dtype).clamp_min(1e-6))
        self.risk_map.risk_stats_initialized.fill_(True)

    @staticmethod
    def _target_boxes_to_pixel_xyxy(target: dict) -> torch.Tensor:
        boxes = target.get("boxes")
        if boxes is None or boxes.numel() == 0:
            return torch.zeros((0, 4), device=target["orig_size"].device)
        xyxy = cxcywh_to_xyxy(boxes.reshape(-1, 4).float()).clamp(0, 1)
        size = target["orig_size"].float().reshape(-1)[:2]
        return xyxy * size.repeat(2)

    @torch.no_grad()
    def update_hard_miss_statistics(self, data_loader, model, postprocessor, device):
        num_types = self.num_tooth_types
        s = self.template_size
        gt_count = torch.zeros((num_types, 1, s, s), device=device, dtype=torch.float64)
        miss_count = torch.zeros((num_types, 1, s, s), device=device, dtype=torch.float64)
        was_training = model.training
        model.eval()
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples, targets=targets)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)
            prepared = self._prepare_targets(targets, device, torch.float32)
            for target, result, prep in zip(targets, results, prepared):
                if prep is None:
                    continue
                tooth_xyxy = prep["boxes_xyxy"]
                tooth_types = prep["types"]
                tooth_rboxes = prep["rboxes"]
                tooth_scores = prep["scores"]
                u_flip = prep["u_flip"]
                a = float(prep.get("aspect", 1.0))
                gt_boxes_px = self._target_boxes_to_pixel_xyxy(target)
                gt_boxes_norm = cxcywh_to_xyxy(target.get("boxes", torch.zeros((0, 4), device=device)).to(device=device, dtype=torch.float32).reshape(-1, 4)).clamp(0, 1)
                num_gt = gt_boxes_norm.shape[0]
                if num_gt == 0:
                    continue
                pred_scores = result["scores"].detach()
                pred_boxes = result["boxes"].detach().float()
                gt_matched = torch.zeros((num_gt,), dtype=torch.bool, device=device)
                if pred_boxes.numel() > 0:
                    order = pred_scores.argsort(descending=True)[:100]
                    pred_boxes = pred_boxes[order]
                    ious = torchvision.ops.box_iou(pred_boxes, gt_boxes_px.float())
                    for i in range(pred_boxes.shape[0]):
                        vals = ious[i].clone()
                        vals[gt_matched] = -1
                        best_iou, best_j = vals.max(dim=0)
                        # COCO-style greedy detection-to-GT matching at IoU=0.50.
                        if best_iou >= 0.50:
                            gt_matched[best_j] = True
                for j, gt in enumerate(gt_boxes_norm):
                    tooth_idx = _assign_gt_to_tooth(gt, tooth_xyxy, tooth_rboxes=tooth_rboxes,
                                                    tooth_scores=tooth_scores, expand_ratio=self.tooth_expanded_box_ratio,
                                                    image_aspect=a)
                    if tooth_idx < 0:
                        continue
                    type_id = int(tooth_types[tooth_idx].item())
                    g = _gt_local_gaussian(gt, tooth_rboxes[tooth_idx], bool(u_flip[tooth_idx].item()), s,
                                           self.risk_gaussian_sigma_min, self.risk_gaussian_sigma_scale, image_aspect=a).to(torch.float64)
                    gt_count[type_id, 0] += g
                    if not bool(gt_matched[j].item()):
                        miss_count[type_id, 0] += g
        if was_training:
            model.train()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(gt_count)
            torch.distributed.all_reduce(miss_count)
        h_epoch = miss_count / (gt_count + float(self.risk_map.hard_miss_confidence_k))
        m = float(self.risk_map.hard_miss_ema_momentum)
        new_h = m * self.risk_map.h_stat.to(device=device, dtype=torch.float64) + (1.0 - m) * h_epoch
        self.risk_map.h_stat.copy_(new_h.to(device=self.risk_map.h_stat.device, dtype=self.risk_map.h_stat.dtype))
