"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .tooth_prior import _assign_gt_to_tooth, _rboxes_for_loss, _corners_to_rbox_uv, _mesial_u_flip, _aspect_from_target
from .tooth_utils import cxcywh_to_xyxy

from ..core import register
import numpy as np


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"


    def _apply_role_aware_cost(self, C: torch.Tensor, outputs: Dict[str, torch.Tensor], targets):
        """Softly encourage query roles without changing the detection losses.

        - tooth queries prefer GTs assigned to their bound tooth.
        - risk queries prefer same-tooth GTs, weighted by selected risk/hard/image priors.
        - context queries prefer same-tooth GTs whose centers are near the tooth local side regions.
        """
        required = ('query_groups', 'query_point_tooth')
        if not all(k in outputs for k in required):
            return C
        groups = outputs['query_groups'].to(device=C.device).long()
        q_tooth = outputs['query_point_tooth'].to(device=C.device).long()
        q_risk = outputs.get('query_risk_prior', torch.zeros(C.shape[:2], device=C.device, dtype=C.dtype)).to(device=C.device, dtype=C.dtype)
        q_hard = outputs.get('query_hard_prior', torch.zeros_like(q_risk)).to(device=C.device, dtype=C.dtype)
        q_img = outputs.get('query_image_risk_prior', torch.zeros_like(q_risk)).to(device=C.device, dtype=C.dtype)
        if groups.shape[:2] != C.shape[:2] or q_tooth.shape[:2] != C.shape[:2]:
            return C

        C = C.clone()
        col_offset = 0
        for b, target in enumerate(targets):
            n_gt = len(target.get('boxes', []))
            if n_gt == 0:
                continue
            col_slice = slice(col_offset, col_offset + n_gt)
            col_offset += n_gt
            if 'tooth_boxes' not in target or target['tooth_boxes'].numel() == 0:
                continue

            device, dtype = C.device, C.dtype
            a = _aspect_from_target(target, default=1.0)
            tooth_boxes = target['tooth_boxes'].to(device=device, dtype=dtype).reshape(-1, 4)
            tooth_xyxy = cxcywh_to_xyxy(tooth_boxes).clamp(0, 1)
            tooth_types = target.get('tooth_types', torch.zeros((tooth_xyxy.shape[0],), device=device, dtype=torch.long)).to(device=device).reshape(-1).long()
            tooth_scores = target.get('tooth_scores', torch.ones((tooth_xyxy.shape[0],), device=device, dtype=dtype)).to(device=device, dtype=dtype).reshape(-1)
            tooth_rboxes = _rboxes_for_loss(target, tooth_xyxy, tooth_types, device, dtype, image_aspect=a)
            midline_boxes = None
            if 'tooth_midline_boxes' in target:
                midline_boxes = target['tooth_midline_boxes'].to(device=device, dtype=dtype).reshape(-1, 4)
                if midline_boxes.shape[0] != tooth_xyxy.shape[0]:
                    midline_boxes = None
            u_flip = _mesial_u_flip(tooth_rboxes, midline_boxes)

            gt_xyxy = box_cxcywh_to_xyxy(target['boxes'].to(device=device, dtype=dtype).reshape(-1, 4)).clamp(0, 1)
            gt_tooth = torch.full((n_gt,), -1, device=device, dtype=torch.long)
            gt_side = torch.zeros((n_gt,), device=device, dtype=torch.bool)
            for j, gt in enumerate(gt_xyxy):
                ti = _assign_gt_to_tooth(gt, tooth_xyxy, tooth_rboxes=tooth_rboxes, tooth_scores=tooth_scores, image_aspect=a)
                if ti < 0:
                    continue
                gt_tooth[j] = int(ti)
                center = ((gt[:2] + gt[2:]) * 0.5).view(1, 2)
                uv = _corners_to_rbox_uv(center, tooth_rboxes[ti], bool(u_flip[ti].item()), image_aspect=a).view(-1)
                gt_side[j] = (uv[0] <= 0.20) | (uv[0] >= 0.80)

            valid_gt = gt_tooth >= 0
            if not valid_gt.any():
                continue
            same_tooth = (q_tooth[b].view(-1, 1) == gt_tooth.view(1, -1)) & valid_gt.view(1, -1)
            g = groups[b].view(-1, 1)
            role_bias = torch.zeros((C.shape[1], n_gt), device=device, dtype=dtype)

            tooth_match = same_tooth & (g == 2)
            role_bias = role_bias + tooth_match.to(dtype) * (-0.20)

            risk_strength = (q_risk[b] + q_hard[b] + q_img[b]).clamp(0, 3).view(-1, 1) / 3.0
            risk_match = same_tooth & (g == 1)
            role_bias = role_bias + risk_match.to(dtype) * (-(0.20 + 0.30 * risk_strength))

            context_match = same_tooth & (g == 3) & gt_side.view(1, -1)
            role_bias = role_bias + context_match.to(dtype) * (-0.25)

            C[b, :, col_slice] = C[b, :, col_slice] + role_bias

        return C

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)
        C = self._apply_role_aware_cost(C, outputs, targets).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # FIXME，RT-DETR, different way to set NaN
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]

        # Compute topk indices
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        # C.copy_(C_original)
        return indices_list
