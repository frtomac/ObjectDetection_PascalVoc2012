from dataclasses import dataclass, field
from math import log
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Type, TypeAlias

from torchvision.ops import box_iou
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import linecache

# from Lib.Data.Annotation import ImageAnnotations

from Lib.Data.Annotation import (
    Annotation,
    Category,
    parse_xml_image_annotations,
    ImageAnnotations,
)

# ImageAnnotations: TypeAlias = list[Annotation]
from Lib.Data.DataVisualizer import show_image_with_annotations
import Transforms

CATEGORIES_IDS = {ctg.value: i for i, ctg in enumerate(Category)}

sizes = [73, 83, 93]


def _default_anchor_widths():
    return sizes  # [3, 15, 7]


def _default_anchor_heights():
    return sizes  # [3, 5, 7]


@dataclass
class GroundTruth:
    bbox_regression: List[float]
    class_confidence: List[float]
    existence_probability: float = 1.0


@dataclass
class GroundTruthGenerationSettings:
    ground_truth_path: Path
    images_path: Path
    num_anchors: int = 3
    num_classes: int = 20
    generate_ground_truth: bool = True
    write_images_and_annotations: bool = True
    "Set to true when original data is meant to be transformed. Not useful otherwise."
    overwrite_existing_ground_truth: bool = True
    anchor_association_iou_threshold: float = 0.35
    grid_rows: int = 3
    grid_cols: int = 3
    input_image_width: int = 180
    input_image_height: int = 180
    anchor_widths: List[int] = field(default_factory=_default_anchor_widths)
    anchor_heights: List[int] = field(default_factory=_default_anchor_heights)
    mu_x: float = 0.0
    mu_y: float = 0.0
    mu_w: float = 0.0
    mu_h: float = 0.0
    sigma_x: float = 0.1
    sigma_y: float = 0.1
    sigma_w: float = 0.2
    sigma_h: float = 0.2

    def __post_init__(self):
        self._verify_and_initialize_gt_generation_settings()

    def _verify_and_initialize_gt_generation_settings(self):
        if self.generate_ground_truth:
            try:
                self.ground_truth_path.mkdir(parents=True, exist_ok=True)
                gt_dir = self.ground_truth_path / "GroundTruth"
                gt_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(
                    f"Ground Truth Generation: You don't have the permission to create {self.ground_truth_path}"
                )
                exit()
            except OSError as e:
                print(f"Ground Truth Generation: An OS error occurred: {e}.")
                exit()

        if self.write_images_and_annotations:  # TODO: update the try statement
            try:
                self.ground_truth_path.mkdir(parents=True, exist_ok=True)
                images_dir = self.ground_truth_path / "Images"
                labels_dir = self.ground_truth_path / "Annotations"
                images_dir.mkdir(parents=True, exist_ok=True)
                labels_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(
                    f"Ground Truth Generation (Images): You don't have the permission to create {self.images_path}"
                )
                exit()
            except OSError as e:
                print(f"Ground Truth Generation (Images): An OS error occurred: {e}.")
                exit()

        assert (
            self.anchor_association_iou_threshold >= 0.0
            and self.anchor_association_iou_threshold <= 1.0
        ), f"GroundTruthSettings: invalid anchor association IoU threshold provided {self.anchor_association_iou_threshold}"

        assert (
            len(self.anchor_widths) == len(self.anchor_heights) == self.num_anchors
        ), f"GroundTruthSettings: num_anchors not aligned with defined anchors."

        # TODO: add verification that anchor widths and heights fit in given the input image and grid size


class PascalVocDataset(Dataset):
    # TODO: write tests! Some class methods are quite tricky and error-prone.
    def __init__(
        self,
        images_path: Path,
        annotations_path: Path,
        ground_truth_generation_settings: GroundTruthGenerationSettings,
        dataset_name: str,
        transform: Type[Transforms.TransformWrapper] = None,
    ):
        if not images_path.exists() or not annotations_path.exists():
            raise FileNotFoundError(
                f"Dataset {dataset_name} initialization failed - provided dataset or annotation files invalid."
            )

        self.images_path: Path = images_path  # JPEG images
        self.annotations_path: Path = annotations_path
        self.transform = transform
        self.filelist: Path = self._create_filelist(dataset_name)
        self.img_labels: List[ImageAnnotations] = self._get_labels()
        self.ground_truth_generation_settings: GroundTruthGenerationSettings = (
            ground_truth_generation_settings
        )

        if self.ground_truth_generation_settings.generate_ground_truth:
            self.ground_truth = self._generate_ground_truth()

        if self.ground_truth_generation_settings.write_images_and_annotations:
            self.write_images()
            self.write_annotations()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # TODO: add functionality to read from transformed folders if available.
        label = self.img_labels[idx]

        img_name = _read_line_from_file(self.filelist, idx)
        if img_name == "":
            raise ValueError(f"Could not get item at index {idx}")

        img_path = self.images_path / f"{img_name[:-1]}.jpg"

        img = Image.open(img_path)
        if self.transform:
            # TODO: labels need to be transformed as well!
            img = self.transform(img)

        return img, label

    def _get_labels(self) -> List[ImageAnnotations]:
        """
        TODO:
            - add doc-string
            - think about returning a generator instead of a list.
        """
        labels = []
        with open(self.filelist, "r") as f:
            for img_name in f:
                if img_name == "":
                    break
                annotations_path = self.annotations_path / f"{img_name[:-1]}.xml"
                img_annotations: ImageAnnotations = parse_xml_image_annotations(
                    annotations_path
                )
                if self.transform:
                    transformed_annotations = []
                    for ann in img_annotations:
                        transformed_annotations.append(
                            self.transform(ann)
                            # self.transform.transform_annotation(ann)
                        )
                    img_annotations = transformed_annotations
                labels.append(img_annotations)

        return labels

    def _create_filelist(self, dataset_name) -> Path:
        """Create a txt file with names of all images in the dataset to fix the ordering.

        Return:
            - None. Creates a txt file in the current project folder.
        """
        filelist_path = (
            Path.cwd() / f"{dataset_name}.txt"
        )  # TODO: Think about how to make this path "standard".

        img_paths = self.images_path.rglob("*.jpg")
        with open(filelist_path, "w+") as f:
            for pth in img_paths:
                img_name = pth.stem
                # img_name = pth
                f.write(str(img_name) + "\n")

        return filelist_path

    def _generate_ground_truth(self):
        """
        TODO: update this doc-string.
        For each object in each image, we calculate ground truth generated for that object.
        """
        negative_anchor_gt = np.zeros(
            1 + 4 + self.ground_truth_generation_settings.num_classes
        )
        total_num_anchors = (
            self.ground_truth_generation_settings.grid_cols
            * self.ground_truth_generation_settings.grid_rows
            * self.ground_truth_generation_settings.num_anchors
        )

        ground_truths = []
        i = 0
        for labels in self.img_labels:  # labels = annotations for one image
            anchors_gt_boxes_with_categories = self.find_anchor_annotation_associations(
                labels
            )
            img_gt = (
                []
            )  # (1 + 4 + num_classes) * list of size num_anchors * grid_rows * grid_cols

            # Calculate the ground truth vector for image
            # TODO: check ground truth generation!
            for anchor_idx in range(total_num_anchors):
                if anchor_idx not in anchors_gt_boxes_with_categories.keys():
                    img_gt.extend(negative_anchor_gt)
                    ground_truths.append(img_gt)
                    continue

                # Anchor is associated with some annotated ground truth box
                associated_anchor = anchors_gt_boxes_with_categories[anchor_idx][0]
                associated_gt_box_and_category = anchors_gt_boxes_with_categories[
                    anchor_idx
                ][1]

                existence_prob = 1.0
                anchor_offset = self.calculate_anchor_offsets(
                    associated_anchor, associated_gt_box_and_category[0]
                )
                class_confidences = list(
                    np.zeros(self.ground_truth_generation_settings.num_classes)
                )
                class_idx = CATEGORIES_IDS[associated_gt_box_and_category[1]]
                class_confidences[class_idx] = 1.0

                img_gt.append(existence_prob)
                img_gt.extend(anchor_offset)
                img_gt.extend(class_confidences)

            ground_truths.append(img_gt)

        return ground_truths

    def write_images(self):
        """Write (transformed) images to a location on the drive.

        The purpose of this function is to store the (transformed) images on the disk to offer the possibility
        of avoiding uneccessary transformation during every run.

        Returns:
            - stores images to self.ground_truth_generation_settings.ground_truth_path / 'Images'.
        """
        with open(self.filelist, "r") as f:
            for img_name in f:
                if img_name == "\n":
                    continue

                img_path = self.images_path / f"{img_name[:-1]}.jpg"
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                    img.save(
                        self.ground_truth_generation_settings.ground_truth_path
                        / "Images"
                        / f"{img_name[:-1]}.jpg"
                    )

    def write_annotations(self):
        pass

    def find_anchor_annotation_associations(self, image_annotations: ImageAnnotations):
        """Associate each ground truth bounding box with an anchor.

        TODO: update this doc-string and describe the algorithm.
        """

        num_gt_boxes = len(image_annotations)
        num_anchors = (
            self.ground_truth_generation_settings.grid_cols
            * self.ground_truth_generation_settings.grid_cols
            * self.ground_truth_generation_settings.num_anchors
        )

        # TODO: is the order of annotations the same in different parts of the code? Can we make it fixed?
        gt_boxes_with_categories = [
            (
                [
                    annotation.bbox.x_min,
                    annotation.bbox.y_min,
                    annotation.bbox.x_max,
                    annotation.bbox.y_max,
                ],
                annotation.category,
            )
            for annotation in image_annotations
        ]
        anchors_with_indices = self._generate_anchors()
        anchors = [
            tuple(anch) for anch in anchors_with_indices.values()
        ]  # list(tuple(anchors_with_indices.values()))
        assert (
            len(anchors) == num_anchors
        ), f"Anchor generation failed: generated {len(anchors)}, expected:{num_anchors}"

        iou_matrix = box_iou(
            torch.tensor(anchors),
            torch.tensor([elem[0] for elem in gt_boxes_with_categories]),
        )

        discarded_col = torch.full((num_anchors,), -1)
        discarded_row = torch.full((num_gt_boxes,), -1)
        anchor_annotation_associations = {}
        for _ in range(num_gt_boxes):
            max_idx = torch.argmax(iou_matrix).item()
            gt_box_idx = max_idx % num_gt_boxes
            anchor_idx = max_idx // num_gt_boxes

            gt_box_with_category = gt_boxes_with_categories[gt_box_idx]
            anchor = anchors_with_indices[anchor_idx]
            anchor_annotation_associations[anchor_idx] = (anchor, gt_box_with_category)

            iou_matrix[anchor_idx, :] = discarded_row
            iou_matrix[:, gt_box_idx] = discarded_col

        return anchor_annotation_associations

    def _generate_anchors(self):
        # TODO: make the ordering of the anchors in the list fixed.
        # TODO: think about creating a class called Box that would hold the upper-left and bottom-right coords of a box
        # to enable type-hinting and also to make the box creation more robust and explicit.
        anchors_with_indices = {}
        num_anchors = self.ground_truth_generation_settings.num_anchors
        for i in range(self.ground_truth_generation_settings.grid_rows):
            for j in range(self.ground_truth_generation_settings.grid_cols):
                tile_idx = i * self.ground_truth_generation_settings.grid_cols + j
                for k, tile_anchor in enumerate(self._generate_tile_anchors(i, j)):
                    anchors_with_indices[tile_idx * num_anchors + k] = tile_anchor

        # img_path = Path(
        #     "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\toy_set\\JPEGImages\\2007_000027.jpg"
        # )
        # img = Image.open(img_path)
        # # im_w, im_h = img.size

        # from Lib.Data.DataVisualizer import (
        #     show_grid_and_anchors_on_image,
        # )

        # anchors = list(anchors_with_indices.values())
        # show_grid_and_anchors_on_image(img, anchors, grid_cols=3, grid_rows=3)

        return anchors_with_indices

    def _generate_tile_anchors(self, i: int, j: int):
        """Generate anchors for a given tile.

        For tile in row i and column j, generate all anchors associated with that tile.

        Args:
            - i (int in range [0, grid_rows)): tile row index
            - j (int in range [0, grid_cols)): tile col index

        Return:
            - List of all anchors associated with that tile in format [x_min, y_min, x_max, y_max].
        """

        # img_path = Path(
        #     "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\toy_set\\JPEGImages\\2007_000027.jpg"
        # )
        # img = Image.open(img_path)
        # im_w, im_h = img.size

        im_w = self.ground_truth_generation_settings.input_image_width
        im_h = self.ground_truth_generation_settings.input_image_height
        rows = self.ground_truth_generation_settings.grid_rows
        cols = self.ground_truth_generation_settings.grid_cols

        tile_w = im_w // cols
        tile_h = im_h // rows
        tile_center_x = int((2 * j + 1) * (tile_w / 2))
        tile_center_y = int((2 * i + 1) * (tile_h / 2))

        anchors = []
        for anchor_w, anchor_h in zip(
            self.ground_truth_generation_settings.anchor_widths,
            self.ground_truth_generation_settings.anchor_heights,
        ):
            anchors.append(
                [
                    tile_center_x - anchor_w // 2,
                    tile_center_y - anchor_h // 2,
                    tile_center_x + anchor_w // 2,
                    tile_center_y + anchor_h // 2,
                ]
            )

        # from Lib.Data.DataVisualizer import (
        #     show_grid_and_anchors_on_image,
        # )

        # show_grid_and_anchors_on_image(img, anchors, grid_cols=3, grid_rows=3)
        return anchors

    def calculate_anchor_offsets(self, anchor, bbox):
        mu_x = self.ground_truth_generation_settings.mu_x
        mu_y = self.ground_truth_generation_settings.mu_y
        mu_w = self.ground_truth_generation_settings.mu_w
        mu_h = self.ground_truth_generation_settings.mu_h
        sigma_x = self.ground_truth_generation_settings.sigma_x
        sigma_y = self.ground_truth_generation_settings.sigma_y
        sigma_w = self.ground_truth_generation_settings.sigma_w
        sigma_h = self.ground_truth_generation_settings.sigma_h

        x_min_idx = 0
        y_min_idx = 1
        x_max_idx = 2
        y_max_idx = 3

        w_gt = bbox[x_max_idx] - bbox[x_min_idx]
        h_gt = bbox[y_max_idx] - bbox[y_min_idx]
        x_gt = bbox[x_min_idx] + (w_gt // 2)
        y_gt = bbox[y_min_idx] + (h_gt // 2)

        w_a = anchor[x_max_idx] - anchor[x_min_idx]
        h_a = anchor[y_max_idx] - anchor[y_min_idx]
        x_a = anchor[x_min_idx] + (w_a // 2)
        y_a = anchor[y_min_idx] + (h_a // 2)

        # TODO: what if divisors are 0?
        dx = (((x_gt - x_a) / w_a) - mu_x) / sigma_x
        dy = (((y_gt - y_a) / h_a) - mu_y) / sigma_y
        dw = (log(w_gt / w_a) - mu_w) / sigma_w
        dh = (log(h_gt / w_a) - mu_h) / sigma_h

        return dx, dy, dw, dh


def _read_line_from_file(filepath: Path, line_index: int) -> str:
    if not filepath.exists():
        raise FileNotFoundError
    if line_index < 0:
        raise ValueError(
            f"_read_line_from_file: invalid line index provided {line_index}"
        )

    res_line = ""
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i != line_index:
                continue
            return line


if __name__ == "__main__":
    ONEIMG_TRAIN_DATA_PATH = Path(
        "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\one_img_set"
    )
    ONEIMG_TRAIN_ANNOTATIONS_PATH = Path(
        "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\one_img_set\\Annotations"
    )

    TOY_TRAIN_IMAGES_PATH = Path(
        "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\toy_set\\JPEGImages"
    )
    TOY_TRAIN_ANNOTATIONS_PATH = Path(
        "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\toy_set\\Annotations"
    )

    TRAIN_ANNOTATIONS_PATH = Path(
        "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\one_img_set\\Annotations"
    )

    gt_settings = GroundTruthGenerationSettings(
        ground_truth_path=Path(
            "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\toy_set\\GroundTruth"
        ),
        images_path=Path(
            "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\toy_set\\JPEGImages"
        ),
        anchor_widths=sizes,
        anchor_heights=sizes,
        generate_ground_truth=True,
        input_image_height=180,
        input_image_width=180,
    )
    transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            Transforms.Resize(
                gt_settings.input_image_height,
                gt_settings.input_image_width,
            ),
        ]
    )
    # transform = Transforms.Resize(
    #     gt_settings.input_image_height,
    #     gt_settings.input_image_width,
    # )
    train_dataset = PascalVocDataset(
        images_path=TOY_TRAIN_IMAGES_PATH,
        annotations_path=TOY_TRAIN_ANNOTATIONS_PATH,
        ground_truth_generation_settings=gt_settings,
        dataset_name="toy_set",
        transform=transform,
    )

    for i in range(0, len(train_dataset)):
        img, label = train_dataset.__getitem__(i)
        show_image_with_annotations(img, label)
    # train_dataset._generate_tile_anchors(0, 2)  # --> works well!
    # train_dataset._generate_anchors()  # --> works well!
