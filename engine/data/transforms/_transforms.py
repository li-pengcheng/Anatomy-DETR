"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()

def _get_box_format_str(boxes, default='XYXY'):
    fmt = getattr(boxes, 'format', None)
    if fmt is None:
        return default

    fmt = getattr(fmt, 'value', fmt)
    fmt = str(fmt).upper()

    # 兼容某些版本可能返回 "BoundingBoxFormat.XYXY"
    if '.' in fmt:
        fmt = fmt.split('.')[-1]

    return fmt


def _rewrap_boxes_like(boxes, kept_raw):
    """
    boxes: 原始 BoundingBoxes，里面带 format 和 canvas_size/spatial_size
    kept_raw: keep 之后的普通 tensor
    """
    spatial_size = getattr(boxes, _boxes_keys[1], None)

    if spatial_size is None:
        # 如果已经没有空间信息，就无法安全 rewrap，只能返回原 tensor
        return kept_raw

    return convert_to_tv_tensor(
        kept_raw,
        key='boxes',
        box_format=_get_box_format_str(boxes),
        spatial_size=spatial_size,
    )


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class SanitizeToothBoxes(nn.Module):
    """Independently sanitize tooth prompts while keeping tooth fields aligned.

    Invalid tooth boxes are removed.  Invalid masks are not removed: they are zeroed so the
    downstream rotated-box builder can safely fall back to the prompt bbox for that tooth.
    """

    def __init__(self, min_size: float = 1.0, mask_min_pixels: int = 12,
                 mask_bbox_min_size: float = 1.0, mask_bbox_clip_margin: float = 0.02) -> None:
        super().__init__()
        self.min_size = float(min_size)
        self.mask_min_pixels = int(mask_min_pixels)
        self.mask_bbox_min_size = float(mask_bbox_min_size)
        self.mask_bbox_clip_margin = float(mask_bbox_clip_margin)

    @staticmethod
    def _tensor_data(value):
        return value.as_subclass(torch.Tensor) if hasattr(value, 'as_subclass') else value

    def _filter_aligned_tensor(self, target: dict, key: str, keep: torch.Tensor):
        if key in target and torch.is_tensor(target[key]) and target[key].shape[:1] == keep.shape:
            target[key] = target[key][keep]

    def _filter_aligned_boxes(self, target: dict, key: str, keep: torch.Tensor):
        if key not in target or not torch.is_tensor(target[key]) or target[key].shape[:1] != keep.shape:
            return
        boxes = target[key]
        raw = self._tensor_data(boxes)
        target[key] = _rewrap_boxes_like(boxes, raw[keep])

    def _filter_corner_boxes(self, target: dict, keep: torch.Tensor):
        """#2 corner boxes are stored as [N*4, 4] (4 per tooth).  Drop the 4 rows of every removed
        tooth by reshaping to [N, 4, 4], indexing with the per-tooth ``keep`` mask, and flattening
        back to [(keep.sum())*4, 4]."""
        key = 'tooth_corner_boxes'
        if key not in target or not torch.is_tensor(target[key]):
            return
        boxes = target[key]
        raw = self._tensor_data(boxes)
        n = int(keep.shape[0])
        if raw.ndim != 2 or raw.shape[0] != n * 4:
            return
        kept = raw.reshape(n, 4, 4)[keep].reshape(-1, 4)
        target[key] = _rewrap_boxes_like(boxes, kept)

    def _zero_invalid_masks(self, masks, boxes_xyxy: torch.Tensor):
        raw_masks = self._tensor_data(masks)
        if raw_masks.ndim != 3 or raw_masks.shape[0] != boxes_xyxy.shape[0]:
            return masks
        if raw_masks.numel() == 0:
            return masks

        h, w = int(raw_masks.shape[-2]), int(raw_masks.shape[-1])
        if h <= 0 or w <= 0:
            return convert_to_tv_tensor(raw_masks.zero_(), key='masks')

        out = raw_masks.clone()
        boxes = boxes_xyxy.to(device=out.device, dtype=torch.float32)
        if boxes.numel() > 0 and float(boxes.max().item()) > 1.5:
            scale = boxes.new_tensor([max(w, 1), max(h, 1), max(w, 1), max(h, 1)])
            boxes_norm = boxes / scale
        else:
            boxes_norm = boxes

        margin = float(self.mask_bbox_clip_margin)
        min_pixels = int(self.mask_min_pixels)
        min_bbox_size = float(self.mask_bbox_min_size)

        for i in range(out.shape[0]):
            ys, xs = (out[i] > 0).nonzero(as_tuple=True)
            valid = xs.numel() >= min_pixels
            if valid:
                mask_w = float(xs.max().item() - xs.min().item() + 1)
                mask_h = float(ys.max().item() - ys.min().item() + 1)
                valid = mask_w >= min_bbox_size and mask_h >= min_bbox_size
            if valid:
                x_norm = (xs.to(torch.float32) + 0.5) / max(w, 1)
                y_norm = (ys.to(torch.float32) + 0.5) / max(h, 1)
                x1, y1, x2, y2 = boxes_norm[i]
                inside = (x_norm >= (x1 - margin).clamp(0, 1)) & \
                         (x_norm <= (x2 + margin).clamp(0, 1)) & \
                         (y_norm >= (y1 - margin).clamp(0, 1)) & \
                         (y_norm <= (y2 + margin).clamp(0, 1))
                valid = int(inside.sum().item()) >= min_pixels
            if not valid:
                out[i].zero_()

        return convert_to_tv_tensor(out, key='masks')

    def _sanitize(self, target: dict):
        if not isinstance(target, dict) or 'tooth_boxes' not in target:
            return target
        boxes = target['tooth_boxes']
        if boxes.numel() == 0:
            target['tooth_scores'] = target.get('tooth_scores', torch.zeros((0,), device=boxes.device, dtype=torch.float32)).reshape(-1)[:0]
            target['tooth_types'] = target.get('tooth_types', torch.zeros((0,), device=boxes.device, dtype=torch.long)).reshape(-1)[:0]
            if 'tooth_masks' in target and torch.is_tensor(target['tooth_masks']):
                target['tooth_masks'] = convert_to_tv_tensor(self._tensor_data(target['tooth_masks'])[:0], key='masks')
            if 'tooth_midline_boxes' in target and torch.is_tensor(target['tooth_midline_boxes']):
                raw_mid = self._tensor_data(target['tooth_midline_boxes'])[:0]
                target['tooth_midline_boxes'] = _rewrap_boxes_like(target['tooth_midline_boxes'], raw_mid)
            if 'tooth_corner_boxes' in target and torch.is_tensor(target['tooth_corner_boxes']):
                raw_cor = self._tensor_data(target['tooth_corner_boxes'])[:0]
                target['tooth_corner_boxes'] = _rewrap_boxes_like(target['tooth_corner_boxes'], raw_cor)
            return target

        raw = self._tensor_data(boxes)
        fmt = getattr(getattr(boxes, 'format', None), 'value', 'XYXY').lower()
        if fmt != 'xyxy':
            xyxy = torchvision.ops.box_convert(raw, in_fmt=fmt, out_fmt='xyxy')
        else:
            xyxy = raw
        keep = (xyxy[:, 2] > xyxy[:, 0] + self.min_size) & (xyxy[:, 3] > xyxy[:, 1] + self.min_size)

        target['tooth_boxes'] = _rewrap_boxes_like(boxes, raw[keep])
        kept_xyxy = xyxy[keep]
        self._filter_aligned_tensor(target, 'tooth_scores', keep)
        self._filter_aligned_tensor(target, 'tooth_types', keep)
        self._filter_aligned_boxes(target, 'tooth_midline_boxes', keep)
        self._filter_corner_boxes(target, keep)
        if 'tooth_masks' in target and torch.is_tensor(target['tooth_masks']):
            raw_masks = self._tensor_data(target['tooth_masks'])
            if raw_masks.shape[:1] == keep.shape:
                target['tooth_masks'] = self._zero_invalid_masks(
                    convert_to_tv_tensor(raw_masks[keep], key='masks'), kept_xyxy)
        return target

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if isinstance(sample, tuple):
            sample = list(sample)
            if len(sample) > 1:
                sample[1] = self._sanitize(sample[1])
            return tuple(sample)
        if isinstance(sample, list):
            if len(sample) > 1:
                sample[1] = self._sanitize(sample[1])
            return sample
        if isinstance(sample, dict):
            return self._sanitize(sample)
        return sample


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt


@register()
class SanitizeBoundingBoxesGT(nn.Module):
    """只清理 target['boxes']（并同步 labels/area/iscrowd），绝不触碰 tooth_* 字段。"""
    def __init__(self, min_size: float = 1.0) -> None:
        super().__init__()
        self.min_size = float(min_size)

    def _sanitize(self, target: dict):
        if not isinstance(target, dict) or 'boxes' not in target:
            return target
        boxes = target['boxes']
        if boxes.numel() == 0:
            return target
        raw = boxes.as_subclass(torch.Tensor) if hasattr(boxes, 'as_subclass') else boxes
        fmt = getattr(getattr(boxes, 'format', None), 'value', 'XYXY').lower()
        xyxy = torchvision.ops.box_convert(raw, in_fmt=fmt, out_fmt='xyxy') if fmt != 'xyxy' else raw
        keep = (xyxy[:, 2] > xyxy[:, 0] + self.min_size) & (xyxy[:, 3] > xyxy[:, 1] + self.min_size)
        target['boxes'] = _rewrap_boxes_like(boxes, raw[keep])
        for k in ('labels', 'area', 'iscrowd'):
            if k in target and torch.is_tensor(target[k]) and target[k].shape[:1] == keep.shape:
                target[k] = target[k][keep]
        return target

    def forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        if isinstance(sample, tuple):
            sample = list(sample)
            if len(sample) > 1:
                sample[1] = self._sanitize(sample[1])
            return tuple(sample)
        if isinstance(sample, list):
            if len(sample) > 1:
                sample[1] = self._sanitize(sample[1])
            return sample
        if isinstance(sample, dict):
            return self._sanitize(sample)
        return sample


@register()
class BuildToothRBoxesFromMasks(nn.Module):
    """Build normalized rotated tooth boxes from final transformed tooth masks.

    This transform must be placed after all geometric augmentations and after
    the final SanitizeToothBoxes, but before ConvertBoxes.

    Input:
        target['tooth_boxes']: pixel-space xyxy BoundingBoxes
        target['tooth_masks']: [N, H, W] Mask

    Output:
        target['tooth_rboxes']: [N, 5], normalized [cx, cy, w, h, angle]
        target['tooth_rbox_valid']: [N], bool
    """

    def __init__(
        self,
        drop_masks: bool = True,
        quantile_low: float = 0.05,
        quantile_high: float = 0.95,
        expand_ratio: float = 1.05,
        min_points: int = 12,
        max_points: int = 512,
        bbox_clip_margin: float = 0.02,
        keep_debug_masks: bool = False,
        upper_tooth_types=None,
        lower_tooth_types=None,
    ) -> None:
        super().__init__()
        self.drop_masks = bool(drop_masks)
        self.quantile_low = float(quantile_low)
        self.quantile_high = float(quantile_high)
        self.expand_ratio = float(expand_ratio)
        self.min_points = int(min_points)
        self.max_points = int(max_points)
        self.bbox_clip_margin = float(bbox_clip_margin)
        self.keep_debug_masks = bool(keep_debug_masks)
        self.upper_tooth_types = upper_tooth_types
        self.lower_tooth_types = lower_tooth_types

    @staticmethod
    def _tensor_data(value):
        return value.as_subclass(torch.Tensor) if hasattr(value, 'as_subclass') else value

    @staticmethod
    def _boxes_to_xyxy(boxes: torch.Tensor):
        raw = boxes.as_subclass(torch.Tensor) if hasattr(boxes, 'as_subclass') else boxes
        if raw.numel() == 0:
            return raw.reshape(-1, 4)

        fmt = getattr(getattr(boxes, 'format', None), 'value', 'XYXY').lower()
        if fmt != 'xyxy':
            raw = torchvision.ops.box_convert(raw, in_fmt=fmt, out_fmt='xyxy')
        return raw.reshape(-1, 4)
    
    @staticmethod
    def _normalize_xyxy(boxes_xyxy: torch.Tensor, boxes, raw_masks=None) -> torch.Tensor:
        """Return normalized xyxy boxes in [0, 1].

        BuildToothRBoxesFromMasks runs before ConvertBoxes, so tooth_boxes are
        normally pixel-space BoundingBoxes. Fallback rboxes must still use the
        same normalized coordinate convention as the normal mask-based path.
        """
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy.reshape(-1, 4).clamp(0, 1)

        h = w = None
        if torch.is_tensor(raw_masks) and raw_masks.ndim >= 2:
            h, w = int(raw_masks.shape[-2]), int(raw_masks.shape[-1])

        if h is None or w is None:
            spatial_size = getattr(boxes, _boxes_keys[1], None)
            if spatial_size is not None:
                h, w = int(spatial_size[0]), int(spatial_size[1])

        if h is None or w is None or h <= 0 or w <= 0:
            return boxes_xyxy.clamp(0, 1)

        if float(boxes_xyxy.max().item()) > 1.5:
            scale = boxes_xyxy.new_tensor([w, h, w, h])
            return (boxes_xyxy / scale).clamp(0, 1)

        return boxes_xyxy.clamp(0, 1)

    @staticmethod
    def _frame_aspect(boxes, raw_masks=None, default: float = 1.0) -> float:
        """Network-frame aspect a = W / H, used to run rbox PCA in an isotropic frame.

        Preference order: the tooth-mask spatial size (it is exactly the network frame the
        rbox is built in), then the box spatial size.  Falls back to ``default`` (1.0 -> no
        correction) only if neither is available.
        """
        h = w = None
        if torch.is_tensor(raw_masks) and raw_masks.ndim >= 2:
            h, w = int(raw_masks.shape[-2]), int(raw_masks.shape[-1])
        if h is None or w is None:
            spatial_size = getattr(boxes, _boxes_keys[1], None)
            if spatial_size is not None:
                h, w = int(spatial_size[0]), int(spatial_size[1])
        if h is None or w is None or h <= 0 or w <= 0:
            return float(default)
        return float(w) / float(h)

    def _build(self, target: dict):
        if not isinstance(target, dict) or 'tooth_boxes' not in target:
            return target

        boxes = target['tooth_boxes']
        boxes_xyxy = self._boxes_to_xyxy(boxes)

        num_tooth = int(boxes_xyxy.shape[0])
        if num_tooth == 0:
            target['tooth_rboxes'] = boxes_xyxy.new_zeros((0, 5), dtype=torch.float32)
            target['tooth_rbox_valid'] = torch.zeros((0,), dtype=torch.bool, device=boxes_xyxy.device)
            target['tooth_rbox_aspect'] = torch.tensor(self._frame_aspect(boxes, None), dtype=torch.float32)
            if self.drop_masks and not self.keep_debug_masks:
                target.pop('tooth_masks', None)
            return target

        tooth_types = target.get(
            'tooth_types',
            torch.zeros((num_tooth,), device=boxes_xyxy.device, dtype=torch.long)
        )

        masks = target.get('tooth_masks', None)
        if masks is None or not torch.is_tensor(masks):
            # No masks: build safe axis-aligned fallback rboxes.
            from ...deim.tooth_prior import _axis_aligned_rboxes_from_xyxy
            a = self._frame_aspect(boxes, None)
            boxes_norm = self._normalize_xyxy(boxes_xyxy, boxes, None)
            target['tooth_rboxes'] = _axis_aligned_rboxes_from_xyxy(
                boxes_norm, tooth_types, image_aspect=a,
                upper_tooth_types=self.upper_tooth_types,
                lower_tooth_types=self.lower_tooth_types,
            ).to(dtype=torch.float32)
            target['tooth_rbox_valid'] = torch.zeros((num_tooth,), dtype=torch.bool, device=boxes_xyxy.device)
            target['tooth_rbox_aspect'] = torch.tensor(a, dtype=torch.float32)
            if self.drop_masks and not self.keep_debug_masks:
                target.pop('tooth_masks', None)
            return target

        raw_masks = self._tensor_data(masks)
        if raw_masks.ndim != 3 or raw_masks.shape[0] != num_tooth:
            from ...deim.tooth_prior import _axis_aligned_rboxes_from_xyxy
            a = self._frame_aspect(boxes, raw_masks)
            boxes_norm = self._normalize_xyxy(boxes_xyxy, boxes, raw_masks)
            target['tooth_rboxes'] = _axis_aligned_rboxes_from_xyxy(
                boxes_norm, tooth_types, image_aspect=a,
                upper_tooth_types=self.upper_tooth_types,
                lower_tooth_types=self.lower_tooth_types,
            ).to(dtype=torch.float32)
            target['tooth_rbox_valid'] = torch.zeros((num_tooth,), dtype=torch.bool, device=boxes_xyxy.device)
            target['tooth_rbox_aspect'] = torch.tensor(a, dtype=torch.float32)
            if self.drop_masks and not self.keep_debug_masks:
                target.pop('tooth_masks', None)
            return target

        a = self._frame_aspect(boxes, raw_masks)
        boxes_norm = self._normalize_xyxy(boxes_xyxy, boxes, raw_masks)

        from ...deim.tooth_prior import _rboxes_from_masks
        rboxes, valid = _rboxes_from_masks(
            raw_masks,
            boxes_norm,
            tooth_types=tooth_types,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            expand_ratio=self.expand_ratio,
            min_points=self.min_points,
            max_points=self.max_points,
            bbox_clip_margin=self.bbox_clip_margin,
            image_aspect=a,
            upper_tooth_types=self.upper_tooth_types,
            lower_tooth_types=self.lower_tooth_types,
        )

        target['tooth_rboxes'] = rboxes.to(dtype=torch.float32)
        target['tooth_rbox_valid'] = valid.to(dtype=torch.bool)
        target['tooth_rbox_aspect'] = torch.tensor(a, dtype=torch.float32)

        if self.drop_masks and not self.keep_debug_masks:
            target.pop('tooth_masks', None)

        return target

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        if isinstance(sample, tuple):
            sample = list(sample)
            if len(sample) > 1:
                sample[1] = self._build(sample[1])
            return tuple(sample)

        if isinstance(sample, list):
            if len(sample) > 1:
                sample[1] = self._build(sample[1])
            return sample

        if isinstance(sample, dict):
            return self._build(sample)

        return sample


@register()
class BuildToothRBoxesFromCorners(nn.Module):
    """#2 Build normalized rotated tooth boxes from the 4 transformed corner points per tooth.

    The corners were computed ONCE at dataset load from the original square-pixel mask (true
    anatomical angle) and carried through every geometric augmentation as tiny boxes.  Here we read
    their transformed centers and re-fit (cx, cy, w, h, angle) with a closed-form 4-point PCA in the
    network-frame aspect-corrected space.  This is orders of magnitude cheaper than re-running mask
    PCA per tooth and removes the need to drag masks through the augmentation pipeline.

    Must be placed after all geometric augmentations and after the final SanitizeToothBoxes, but
    before ConvertBoxes (so the corner/tooth boxes are still pixel-space xyxy).

    Input:
        target['tooth_corner_boxes']: [N*4, 4] pixel xyxy BoundingBoxes (4 corners per tooth)
        target['tooth_boxes']:        [N, 4]   pixel xyxy BoundingBoxes (for fallback / frame size)
    Output:
        target['tooth_rboxes']:       [N, 5]   normalized [cx, cy, w, h, angle]
        target['tooth_rbox_valid']:   [N]      bool
        target['tooth_rbox_aspect']:  scalar   a = W / H of the network frame
    """

    def __init__(self, drop_corners: bool = True, upper_tooth_types=None, lower_tooth_types=None) -> None:
        super().__init__()
        self.drop_corners = bool(drop_corners)
        self.upper_tooth_types = upper_tooth_types
        self.lower_tooth_types = lower_tooth_types

    @staticmethod
    def _tensor_data(value):
        return value.as_subclass(torch.Tensor) if hasattr(value, 'as_subclass') else value

    @staticmethod
    def _xyxy(boxes):
        raw = boxes.as_subclass(torch.Tensor) if hasattr(boxes, 'as_subclass') else boxes
        if raw.numel() == 0:
            return raw.reshape(-1, 4)
        fmt = getattr(getattr(boxes, 'format', None), 'value', 'XYXY').lower()
        if fmt != 'xyxy':
            raw = torchvision.ops.box_convert(raw, in_fmt=fmt, out_fmt='xyxy')
        return raw.reshape(-1, 4)

    @staticmethod
    def _frame_hw(boxes, default_hw=None):
        spatial_size = getattr(boxes, _boxes_keys[1], None)
        if spatial_size is not None:
            return int(spatial_size[0]), int(spatial_size[1])
        return default_hw

    def _build(self, target: dict):
        if not isinstance(target, dict) or 'tooth_boxes' not in target:
            return target

        boxes = target['tooth_boxes']
        boxes_xyxy = self._xyxy(boxes)
        num_tooth = int(boxes_xyxy.shape[0])

        hw = self._frame_hw(boxes, None)
        if hw is None and 'tooth_corner_boxes' in target:
            hw = self._frame_hw(target['tooth_corner_boxes'], None)
        h, w = (hw if hw is not None else (1, 1))
        a = float(w) / float(max(h, 1))

        tooth_types = target.get('tooth_types',
                                 torch.zeros((num_tooth,), device=boxes_xyxy.device, dtype=torch.long))

        from ...deim.tooth_prior import _fit_rbox_from_corners, _axis_aligned_rboxes_from_xyxy

        def _normalize_boxes(bx):
            if bx.numel() and float(bx.max().item()) > 1.5:
                return (bx / bx.new_tensor([w, h, w, h])).clamp(0, 1)
            return bx.clamp(0, 1)

        # Always compute the axis-aligned fallback so invalid corner fits degrade gracefully.
        boxes_norm = _normalize_boxes(boxes_xyxy)
        fallback = _axis_aligned_rboxes_from_xyxy(
            boxes_norm, tooth_types, image_aspect=a,
            upper_tooth_types=self.upper_tooth_types, lower_tooth_types=self.lower_tooth_types,
        ).to(dtype=torch.float32) if num_tooth > 0 else boxes_xyxy.new_zeros((0, 5), dtype=torch.float32)

        corners = target.get('tooth_corner_boxes', None)
        valid = torch.zeros((num_tooth,), dtype=torch.bool, device=boxes_xyxy.device)
        rboxes = fallback.clone()

        if num_tooth > 0 and corners is not None and torch.is_tensor(corners):
            craw = self._xyxy(corners)
            if craw.ndim == 2 and craw.shape[0] == num_tooth * 4:
                # centers of the 4 tiny corner boxes -> [N, 4, 2] normalized network coords
                cx = (craw[:, 0] + craw[:, 2]) * 0.5
                cy = (craw[:, 1] + craw[:, 3]) * 0.5
                centers_px = torch.stack([cx, cy], dim=-1).reshape(num_tooth, 4, 2)
                centers_norm = centers_px / centers_px.new_tensor([w, h])
                centers_norm = centers_norm.clamp(0, 1)
                fitted, fit_valid = _fit_rbox_from_corners(
                    centers_norm, tooth_types=tooth_types, image_aspect=a,
                    upper_tooth_types=self.upper_tooth_types, lower_tooth_types=self.lower_tooth_types,
                )
                fitted = fitted.to(dtype=rboxes.dtype, device=rboxes.device)
                fit_valid = fit_valid.to(device=rboxes.device)
                rboxes = torch.where(fit_valid.unsqueeze(-1), fitted, rboxes)
                valid = fit_valid

        target['tooth_rboxes'] = rboxes.to(dtype=torch.float32)
        target['tooth_rbox_valid'] = valid.to(dtype=torch.bool)
        target['tooth_rbox_aspect'] = torch.tensor(a, dtype=torch.float32)

        if self.drop_corners:
            target.pop('tooth_corner_boxes', None)
        target.pop('tooth_masks', None)
        return target

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if isinstance(sample, tuple):
            sample = list(sample)
            if len(sample) > 1:
                sample[1] = self._build(sample[1])
            return tuple(sample)
        if isinstance(sample, list):
            if len(sample) > 1:
                sample[1] = self._build(sample[1])
            return sample
        if isinstance(sample, dict):
            return self._build(sample)
        return sample