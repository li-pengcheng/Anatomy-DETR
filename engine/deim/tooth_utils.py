"""
Tooth-aware utilities for Stage 1+ risk-guided query selection.
"""

import torch


def _weighted_median(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.5)
    if weights is None or weights.numel() != values.numel() or float(weights.sum()) <= 0:
        return values.median()
    order = torch.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order].clamp(min=0)
    cdf = weights_sorted.cumsum(0)
    cutoff = weights_sorted.sum() * 0.5
    return values_sorted[torch.searchsorted(cdf, cutoff).clamp(max=values_sorted.numel() - 1)]


def _kmeans_1d_two_clusters(values: torch.Tensor, max_iter: int = 20):
    """Small dependency-free 1D 2-means used for upper/lower tooth rows."""
    if values.numel() < 2 or torch.isclose(values.min(), values.max()):
        return None
    centers = torch.stack([values.min(), values.max()]).to(values)
    labels = torch.zeros_like(values, dtype=torch.long)
    for _ in range(max_iter):
        dist = torch.abs(values[:, None] - centers[None, :])
        new_labels = dist.argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(2):
            if (labels == k).any():
                centers[k] = values[labels == k].mean()
    if not (labels == 0).any() or not (labels == 1).any():
        return None
    upper_cluster = torch.argmin(centers)
    return labels == upper_cluster


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1)], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def expand_xyxy_boxes(boxes: torch.Tensor, ratio: float = 1.1) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    cxcywh = xyxy_to_cxcywh(boxes)
    cxcywh[..., 2:] = cxcywh[..., 2:] * ratio
    return cxcywh_to_xyxy(cxcywh)


def classify_tooth_types_from_boxes(boxes_xyxy: torch.Tensor, scores: torch.Tensor = None) -> torch.Tensor:
    """
    Coarsely classify panoramic X-ray tooth boxes into four hard types:
      0 upper anterior, 1 upper posterior, 2 lower anterior, 3 lower posterior.

    The classification is intentionally heuristic and runs before augmentation.
    """
    boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
    n = boxes.shape[0]
    if n == 0:
        return torch.zeros((0,), dtype=torch.long)

    scores = torch.ones((n,), dtype=torch.float32) if scores is None or len(scores) != n else torch.as_tensor(scores, dtype=torch.float32)
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    area = w * h

    upper_mask = _kmeans_1d_two_clusters(cy)
    if upper_mask is None:
        med_y = cy.median()
        upper_mask = cy < med_y
        if not upper_mask.any() or upper_mask.all():
            upper_mask = cy <= med_y

    mid_x = _weighted_median(cx, scores)
    types = torch.zeros((n,), dtype=torch.long)

    for is_upper, row_offset in [(True, 0), (False, 2)]:
        row_mask = upper_mask if is_upper else ~upper_mask
        idx = torch.nonzero(row_mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue

        d = torch.abs(cx[idx] - mid_x)
        d_max = d.max().clamp(min=1e-6)
        center_score = 1.0 - d / d_max

        aspect = h[idx] / w[idx]
        aspect_min, aspect_max = aspect.min(), aspect.max()
        aspect_score = (aspect - aspect_min) / (aspect_max - aspect_min).clamp(min=1e-6)

        med_w = w[idx].median().clamp(min=1e-6)
        med_area = area[idx].median().clamp(min=1e-6)
        rel_w = w[idx] / med_w
        rel_area = area[idx] / med_area
        size_raw = 1.0 / (0.5 * rel_w + 0.5 * rel_area).clamp(min=1e-6)
        size_score = (size_raw - size_raw.min()) / (size_raw.max() - size_raw.min()).clamp(min=1e-6)

        anterior_score = 0.65 * center_score + 0.25 * aspect_score + 0.10 * size_score
        if idx.numel() <= 2:
            threshold = anterior_score.mean()
        else:
            threshold = anterior_score.median()
        anterior = anterior_score >= threshold
        types[idx] = row_offset + (~anterior).long()

    return types.long()
