import numpy as np
import torch
from torchvision.ops import box_iou

from Lib.GroundTruthCalculations import _generate_tile_anchors, find_box_associations
from Lib.Data.Annotation import ImageAnnotations, ImageDetections


# _generate_tile_anchors(
#     tile_row: int,
#     tile_col: int,
#     img_w: int,
#     img_h: int,
#     grid_rows: int,
#     grid_cols: int,
#     anchor_widths: List,
#     anchor_heights: List,
# )

"""
TODO

Localization metrics
- precision, recall, f1
- precision-recall curve
- average IoU

Classification metrics
- confusion matrix

Train metrics
- loss/epoch curve
"""


def create_detections(model_output, conf_threshold: float = 0.8) -> ImageDetections:
    pass


def _apply_bbox_offsets(bbox, offsets):
    pass


def categorize_predictions(annotations: ImageAnnotations, detections: ImageDetections):
    gt_boxes = [ann.bbox for ann in annotations]
    det_boxes = [det.bbox for det in detections]

    total_num_detections = len(det_boxes)
    total_num_gt_boxes = len(gt_boxes)

    associations = find_box_associations(gt_boxes, det_boxes)
    num_associations = len(associations)

    assert (
        num_associations <= total_num_gt_boxes
    ), "Number of associations is greater than the num of gt boxes."

    fn = total_num_gt_boxes - num_associations
    tp = num_associations
    fp = total_num_detections - num_associations

    return tp, fp, fn


def calculate_localization_statistics(
    annotations: ImageAnnotations, detections: ImageDetections
):
    tp, fp, fn = categorize_predictions(annotations, detections)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)


def _calculate_batch_statistics(
    gt_offsets,  # batch_size * 4 * grid_rows * grid_cols * num_anchors
    gt_classes,  # batch_size * num_classes * grid_rows * grid_cols * num_anchors
    out_offsets,
    out_classes,
    grid_rows,
    grid_cols,
    num_anchors,
    num_classes,
    img_w,
    img_h,
    anchor_widths,
    anchor_heights,
    iou_threshold=0.5,
):
    for _ in range(len(gt_offsets)):  # iterate over batch
        for i in range(grid_rows * grid_cols * num_anchors):
            gt_cls = torch.argmax(
                gt_classes[i * num_classes : (i + 1) * num_classes]
            ).item()
            out_cls = torch.argmax(
                out_classes[i * num_classes : (i + 1) * num_classes]
            ).item()

            # anchor_idx = tile_row * grid_cols + tile_col + k
            k = i % num_anchors
            tile_row = i // grid_cols
            tile_col = i - i * grid_cols - k
            anchor = _generate_tile_anchors(
                tile_row,
                tile_col,
                img_w,
                img_h,
                grid_rows,
                grid_cols,
                anchor_widths,
                anchor_heights,
            )[k]

            gt_off = gt_offsets[i * 4 : (i + 1) * 4]
            out_off = out_offsets[i * 4 : (i + 1) * 4]
            gt_box = torch.tensor(_apply_bbox_offsets(anchor, gt_off))
            out_box = torch.tensor(_apply_bbox_offsets(anchor, out_off))

            iou = box_iou(gt_box, out_box).item()
            if iou > iou_threshold:
                pass
            else:
                pass
