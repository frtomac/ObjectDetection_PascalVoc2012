from torchvision.ops import complete_box_iou
import torch
from typing import List


def find_box_associations(boxes1, boxes2):
    num_boxes1 = len(boxes1)
    num_boxes2 = len(boxes2)

    iou_matrix = complete_box_iou(
        torch.tensor(boxes1),
        torch.tensor(boxes2),
    )

    discarded_col = torch.full((num_boxes2,), -1)
    discarded_row = torch.full((num_boxes1,), -1)

    associations = {}
    for _ in range(num_boxes2):
        max_idx = torch.argmax(iou_matrix).item()
        box2_idx = max_idx % num_boxes2
        box1_idx = max_idx // num_boxes2

        box2 = boxes2[box2_idx]
        box1 = boxes1[box1_idx]
        associations[box1_idx] = (box1, box2)

        iou_matrix[box1_idx, :] = discarded_row
        iou_matrix[:, box2_idx] = discarded_col

    return associations


def _generate_tile_anchors(
    tile_row: int,
    tile_col: int,
    img_w: int,
    img_h: int,
    grid_rows: int,
    grid_cols: int,
    anchor_widths: List,
    anchor_heights: List,
) -> List:
    """Generate anchors for a given tile.

    For tile in row i and column j, generate all anchors associated with that tile.

    Args:
        - i (int in range [0, grid_rows)): tile row index
        - j (int in range [0, grid_cols)): tile col index

    Return:
        - List of all anchors associated with that tile in format [x_min, y_min, x_max, y_max].
    """

    # im_w = self.ground_truth_generation_settings.input_image_width
    # im_h = self.ground_truth_generation_settings.input_image_height
    # rows = self.ground_truth_generation_settings.grid_rows
    # cols = self.ground_truth_generation_settings.grid_cols

    tile_w = img_w // grid_cols
    tile_h = img_h // grid_rows
    tile_center_x = int((2 * tile_col + 1) * (tile_w / 2))
    tile_center_y = int((2 * tile_row + 1) * (tile_h / 2))

    anchors = []
    for anchor_w, anchor_h in zip(
        anchor_widths,
        anchor_heights,
    ):
        anchors.append(
            [
                tile_center_x - anchor_w // 2,
                tile_center_y - anchor_h // 2,
                tile_center_x + anchor_w // 2,
                tile_center_y + anchor_h // 2,
            ]
        )

    return anchors
