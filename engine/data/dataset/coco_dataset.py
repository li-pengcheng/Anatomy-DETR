"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import json
from collections import defaultdict

import torch
import torch.utils.data

import torchvision

from PIL import Image
import faster_coco_eval
import faster_coco_eval.core.mask as coco_mask
from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

torchvision.disable_beta_transforms_warning()
faster_coco_eval.init_as_pycocotools()
Image.MAX_IMAGE_PIXELS = None

__all__ = ['CocoDetection']


def _is_flat_polygon(segmentation):
    return isinstance(segmentation, list) and all(isinstance(v, (int, float)) for v in segmentation)


def _decode_single_segmentation(segmentation, height, width):
    """Decode one COCO polygon/RLE segmentation to a binary mask.

    The external tooth detector may output polygons, uncompressed RLE, or compressed RLE.
    If decoding fails or segmentation is missing, return an empty mask so the caller can
    safely fall back to the original prompt bbox.
    """
    empty = torch.zeros((height, width), dtype=torch.uint8)
    if segmentation in (None, '', 'null'):
        return empty

    try:
        if isinstance(segmentation, dict):
            # Compressed RLE can be decoded directly; uncompressed RLE needs frPyObjects.
            if isinstance(segmentation.get('counts', None), list):
                rle = coco_mask.frPyObjects(segmentation, height, width)
            else:
                rle = segmentation
            mask = coco_mask.decode(rle)
        elif isinstance(segmentation, list):
            polygons = [segmentation] if _is_flat_polygon(segmentation) else segmentation
            if len(polygons) == 0:
                return empty
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
        else:
            return empty

        if len(mask.shape) == 3:
            mask = mask.any(axis=2)
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        if mask.ndim != 2:
            return empty
        return mask.to(dtype=torch.uint8)
    except Exception:
        return empty

@register()
class CocoDetection(torchvision.datasets.CocoDetection, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category', 'num_classes']

    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False,
                 tooth_json_file=None, tooth_score_thr=0.5, max_tooth_boxes=None,
                 tooth_num_types=6, tooth_category_offset=None, num_classes=80,
                 validate_category_ids=True, tooth_rbox_source='corners',
                 upper_tooth_types=(0, 2, 4), lower_tooth_types=(1, 3, 5)):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.tooth_json_file = tooth_json_file
        self.use_tooth_predictions = tooth_json_file not in (None, '', 'null')
        self.tooth_score_thr = float(tooth_score_thr)
        self.max_tooth_boxes = max_tooth_boxes
        self.tooth_num_types = int(tooth_num_types)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.validate_category_ids = bool(validate_category_ids)
        # #2 'corners': run mask-PCA ONCE at load in the original (square-pixel) frame, store the rbox as
        # 4 tiny corner boxes that follow every geometric augmentation, and drop the heavy mask. 'masks':
        # legacy path that carries masks through augmentation and fits the rbox at the end (slower, more
        # memory).  'corners' is both faster and geometrically more correct (true anatomical angle).
        self.tooth_rbox_source = str(tooth_rbox_source)
        self.upper_tooth_types = tuple(int(v) for v in upper_tooth_types)
        self.lower_tooth_types = tuple(int(v) for v in lower_tooth_types)
        # None means auto: 0-based if any category_id is 0, otherwise 1-based -> internal 0-based.
        self.tooth_category_offset = None if tooth_category_offset in (None, '', 'auto') else int(tooth_category_offset)
        self._validate_detection_category_ids()
        self.tooth_predictions = self._load_tooth_predictions(tooth_json_file)

    def _normalize_tooth_category(self, category_id):
        raw_category_id = int(category_id)
        if self.tooth_category_offset is None:
            category_id = raw_category_id - getattr(self, '_auto_tooth_category_offset', 0)
        else:
            category_id = raw_category_id - self.tooth_category_offset
        if self.validate_category_ids:
            assert 0 <= category_id < self.tooth_num_types, (
                f'Invalid prompt tooth category_id={raw_category_id} after offset -> {category_id}; '
                f'expected [0, {self.tooth_num_types - 1}]. '
                f'Check tooth_category_offset and the 0-based 6-class prompt ids.'
            )
            return category_id
        return max(0, min(category_id, self.tooth_num_types - 1))

    def _validate_detection_category_ids(self):
        if not self.validate_category_ids or self.remap_mscoco_category or self.num_classes is None:
            return
        annotations = self.coco.dataset.get('annotations', []) if hasattr(self, 'coco') else []
        bad = []
        for obj in annotations:
            if not isinstance(obj, dict):
                continue
            if obj.get('iscrowd', 0) != 0:
                continue
            if 'category_id' not in obj:
                continue
            cid = int(obj['category_id'])
            if cid < 0 or cid >= self.num_classes:
                bad.append(cid)
                if len(bad) >= 8:
                    break
        assert not bad, (
            f'Invalid caries annotation category_id values {bad}; with num_classes={self.num_classes} '
            f'and remap_mscoco_category=False, detection labels must be 0-based in [0, {self.num_classes - 1}].'
        )

    def _validate_prompt_category_ids(self, category_ids):
        if not self.validate_category_ids:
            return
        if not category_ids:
            return
        offset = getattr(self, '_auto_tooth_category_offset', 0) if self.tooth_category_offset is None else self.tooth_category_offset
        bad = []
        for cid in category_ids:
            norm = int(cid) - int(offset)
            if norm < 0 or norm >= self.tooth_num_types:
                bad.append(int(cid))
                if len(bad) >= 8:
                    break
        assert not bad, (
            f'Invalid prompt tooth category_id values {bad}; after offset={offset}, prompt ids must be '
            f'0-based in [0, {self.tooth_num_types - 1}].'
        )

    def _load_tooth_predictions(self, tooth_json_file):
        if tooth_json_file in (None, '', 'null'):
            return {}
        with open(tooth_json_file, 'r') as f:
            data = json.load(f)

        # Accept either a COCO-style dict with an annotations field or a plain list of predictions.
        annotations = data.get('annotations', data) if isinstance(data, dict) else data

        category_ids = [int(obj['category_id']) for obj in annotations if isinstance(obj, dict) and 'category_id' in obj]
        if self.tooth_category_offset is None:
            # External YOLO-style prompts are often 0-based; COCO-style prompts are often 1-based.
            # Auto is conservative: only subtract 1 when the json clearly looks 1-based.
            if any(cid == 0 for cid in category_ids):
                self._auto_tooth_category_offset = 0
            elif category_ids and max(category_ids) == self.tooth_num_types:
                self._auto_tooth_category_offset = 1
            elif len(set(category_ids)) == self.tooth_num_types and min(category_ids) == 1:
                self._auto_tooth_category_offset = 1
            else:
                self._auto_tooth_category_offset = 0

        self._validate_prompt_category_ids(category_ids)

        image_to_records = defaultdict(list)
        for obj in annotations:
            if not isinstance(obj, dict) or 'image_id' not in obj or 'bbox' not in obj:
                continue
            score = float(obj.get('score', 1.0))
            if score < self.tooth_score_thr:
                continue
            image_to_records[int(obj['image_id'])].append(obj)

        if self.max_tooth_boxes is not None:
            max_boxes = int(self.max_tooth_boxes)
            for image_id, records in image_to_records.items():
                records.sort(key=lambda x: float(x.get('score', 1.0)), reverse=True)
                image_to_records[image_id] = records[:max_boxes]

        return dict(image_to_records)

    def _get_tooth_target(self, image_id, width, height):
        records = self.tooth_predictions.get(int(image_id), [])
        boxes, scores, types, masks, midline_boxes = [], [], [], [], []
        height_i, width_i = int(height), int(width)
        # One per-tooth source-image midline anchor.  It is represented as a tiny
        # bounding box so all torchvision geometric transforms, including Mosaic,
        # RandomHorizontalFlip, RandomAffine and Resize, keep it synchronized with
        # the corresponding tooth prompt.
        mid_x = float(width) * 0.5
        mid_eps = max(float(width), 1.0) * 0.001
        mid_y1, mid_y2 = 0.0, float(height)
        for obj in records:
            x, y, w, h = obj['bbox']
            x1 = max(0.0, min(float(x), float(width)))
            y1 = max(0.0, min(float(y), float(height)))
            x2 = max(0.0, min(float(x + w), float(width)))
            y2 = max(0.0, min(float(y + h), float(height)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            scores.append(float(obj.get('score', 1.0)))
            types.append(self._normalize_tooth_category(obj.get('category_id', 0)))
            masks.append(_decode_single_segmentation(obj.get('segmentation', None), height_i, width_i))
            midline_boxes.append([
                max(0.0, mid_x - mid_eps),
                mid_y1,
                min(float(width), mid_x + mid_eps),
                mid_y2,
            ])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
        types = torch.as_tensor(types, dtype=torch.long).reshape(-1)
        midline_boxes = torch.as_tensor(midline_boxes, dtype=torch.float32).reshape(-1, 4)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height_i, width_i), dtype=torch.uint8)

        # #2 corner-based rbox: run the mask PCA ONCE here, in the ORIGINAL square-pixel frame, and
        # store each tooth's 4 rbox corners as tiny pixel boxes that will follow every geometric
        # augmentation.  The (heavy) mask is then no longer needed downstream and is dropped at load.
        corner_boxes = self._tooth_corner_boxes(boxes, masks, types, width_i, height_i)
        return boxes, scores, types, masks, midline_boxes, corner_boxes

    def _tooth_corner_boxes(self, boxes_xyxy_px, masks, types, width_i, height_i):
        """Return [N*4, 4] pixel-xyxy tiny corner boxes (4 per tooth) for the corner-rbox path.

        The rbox is estimated from the original-pixel mask (square pixels -> aspect = W/H) so the
        recovered angle is the true anatomical angle; its 4 corners are then emitted as tiny boxes.
        """
        n = int(boxes_xyxy_px.shape[0])
        if n == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        from ...deim.tooth_prior import _rboxes_from_masks, _rbox_to_corners, _axis_aligned_rboxes_from_xyxy
        a = float(width_i) / float(max(height_i, 1))
        boxes_norm = (boxes_xyxy_px / boxes_xyxy_px.new_tensor([max(width_i, 1), max(height_i, 1),
                                                                max(width_i, 1), max(height_i, 1)])).clamp(0, 1)
        if torch.is_tensor(masks) and masks.ndim == 3 and masks.shape[0] == n:
            rboxes, _valid = _rboxes_from_masks(masks, boxes_norm, tooth_types=types, image_aspect=a,
                                                upper_tooth_types=self.upper_tooth_types,
                                                lower_tooth_types=self.lower_tooth_types)
        else:
            rboxes = _axis_aligned_rboxes_from_xyxy(boxes_norm, types, image_aspect=a,
                                                    upper_tooth_types=self.upper_tooth_types,
                                                    lower_tooth_types=self.lower_tooth_types)
        corners_norm = _rbox_to_corners(rboxes, image_aspect=a)          # [N, 4, 2] normalized
        cx = (corners_norm[..., 0] * float(width_i))                     # [N, 4] pixel
        cy = (corners_norm[..., 1] * float(height_i))
        eps = 0.5
        corner_boxes = torch.stack([cx - eps, cy - eps, cx + eps, cy + eps], dim=-1)  # [N, 4, 4]
        return corner_boxes.reshape(-1, 4).to(torch.float32)             # [N*4, 4]

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        if self.validate_category_ids and self.num_classes is not None and 'labels' in target and target['labels'].numel() > 0:
            mn = int(target['labels'].min().item())
            mx = int(target['labels'].max().item())
            assert mn >= 0 and mx < self.num_classes, (
                f'Invalid detection labels in image_id={int(image_id)}: min={mn}, max={mx}, '
                f'expected 0 <= label < num_classes ({self.num_classes}).'
            )

        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if self.use_tooth_predictions:
            tooth_boxes, tooth_scores, tooth_types, tooth_masks, tooth_midline_boxes, tooth_corner_boxes = \
                self._get_tooth_target(image_id, *image.size)
            target['tooth_boxes'] = convert_to_tv_tensor(tooth_boxes, key='boxes', spatial_size=image.size[::-1])
            target['tooth_scores'] = tooth_scores
            if self.validate_category_ids and tooth_types.numel() > 0:
                mn = int(tooth_types.min().item())
                mx = int(tooth_types.max().item())
                assert mn >= 0 and mx < self.tooth_num_types, (
                    f'Invalid prompt tooth types in image_id={int(image_id)}: min={mn}, max={mx}, '
                    f'expected 0 <= type < tooth_num_types ({self.tooth_num_types}).'
                )
            target['tooth_types'] = tooth_types
            target['tooth_midline_boxes'] = convert_to_tv_tensor(
                tooth_midline_boxes, key='boxes', spatial_size=image.size[::-1])
            if self.tooth_rbox_source == 'corners':
                # Corner path: carry only the 4 tiny corner boxes per tooth; drop the heavy mask here.
                target['tooth_corner_boxes'] = convert_to_tv_tensor(
                    tooth_corner_boxes, key='boxes', spatial_size=image.size[::-1])
            else:
                # Legacy mask path: carry masks through augmentation; rbox fitted at the end.
                target['tooth_masks'] = convert_to_tv_tensor(tooth_masks, key='masks')

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return image, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        s += f' tooth_json_file: {self.tooth_json_file}\n'
        s += f' tooth_score_thr: {self.tooth_score_thr}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s

    @property
    def categories(self, ):
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
