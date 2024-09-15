from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List

from torchvision.ops import box_iou
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import linecache

from Lib.Data.Annotation import Annotation, Category, parse_xml_image_annotations
from Lib.Data.DataVisualizer import show_image_with_annotations

CATEGORIES_IDS = {ctg.value: i for i, ctg in enumerate(Category)}


@dataclass
class GroundTruth:
    bbox_regression: List[float, float, float, float]
    class_confidence: List[int]
    existence_probability: int = 1


@dataclass
class GroundTruthGenerationSettings:
    ground_truth_path: Path
    num_anchors: int = 3
    num_classes: int = 20
    generate_ground_truth: bool = True
    overwrite_existing_ground_truth: bool = True
    anchor_association_iou_threshold: float = 0.35
    grid_rows: int = 9
    grid_cols: int = 9
    input_image_width: int = 180
    input_image_height: int = 180
    anchor_widths: List[int] = [7, 9, 11]
    anchor_heights: List[int] = [7, 9, 11]

    def __post_init__(self):
        self._verify_and_initialize_gt_generation_settings()

    def _verify_and_initialize_gt_generation_settings(self):
        if self.generate_ground_truth:
            try:
                self.ground_truth_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(
                    f"Ground Truth Generation: You don't have the permission to create {self.ground_truth_path}"
                )
                exit()
            except OSError as e:
                print(f"Ground Truth Generation: An OS error occurred: {e}.")
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
    def __init__(
        self,
        dataset_path: Path,
        annotations_path: Path,
        ground_truth_generation_settings: GroundTruthGenerationSettings,
        dataset_name: str,
        transform=None,
    ):
        if not dataset_path.exists() or not annotations_path.exists():
            raise FileNotFoundError

        self.dataset_path: Path = dataset_path  # JPEG images
        self.annotations_path: Path = annotations_path
        self.filelist: Path = self._create_filelist(dataset_name)
        self.img_labels: List[List[Annotation]] = self._get_labels()
        self.transform = transform
        self.ground_truth_generation_settings: GroundTruthGenerationSettings = (
            ground_truth_generation_settings
        )

        if self.ground_truth_generation_settings.generate_ground_truth:
            self._generate_ground_truth()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels[idx]

        # linecache.clearcache()
        # img_name = linecache.getline(str(self.filelist), idx)
        img_name = _read_line_from_file(self.filelist, idx)
        if img_name == "":
            raise ValueError(f"Could not get item at index {idx}")

        img_path = self.dataset_path / f"{img_name[:-1]}.jpg"

        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        return img, label

    def _get_labels(self) -> List[List[Annotation]]:
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
                labels.append(parse_xml_image_annotations(annotations_path))

        return labels

    def _create_filelist(self, dataset_name) -> Path:
        """Create a txt file with names of all images in the dataset to fix the ordering.

        Return:
            - None. Creates a txt file in the current project folder.
        """
        filelist_path = (
            Path.cwd() / f"{dataset_name}.txt"
        )  # TODO: Think about how to make this path "standard".

        img_paths = self.dataset_path.rglob("*.jpg")
        with open(filelist_path, "w+") as f:
            for pth in img_paths:
                img_name = pth.stem
                # img_name = pth
                f.write(str(img_name) + "\n")

        return filelist_path

    def _generate_ground_truth(self) -> List[List[GroundTruth]]:
        """
        TODO: update this doc-string.
        For each object in each image, we need to calculate ground truth generated for that object.
        """
        ground_truths = []
        for labels in self.img_labels:
            # # Existence probability ground truth
            # existence_probability_gt = 1

            # # Class confidence ground truth
            # class_confidence_gt = list(
            #     np.zeros(self.ground_truth_generation_settings.num_classes)
            # )
            # class_idx = CATEGORIES_IDS[annotation.category]
            # class_confidence_gt[class_idx] = 1

            # Bounding box regression ground truth
            associations = self.find_anchor_box_associations(labels)

        return ground_truths

    def find_anchor_box_associations(self, labels: List[Annotation]):
        object_anchor_associations = []
        iou_threshold = (
            self.ground_truth_generation_settings.anchor_association_iou_threshold
        )
        for label in labels:  # annotations for one image
            for annotation in label:
                bbox = [
                    annotation.bbox.xmin,
                    annotation.bbox.ymin,
                    annotation.bbox.xmax,
                    annotation.bbox.ymax,
                ]

                associations = []
                for i in range(self.ground_truth_generation_settings.grid_rows):
                    for j in range(self.ground_truth_generation_settings.grid_cols):
                        tile_anchors = self._generate_tile_anchors(i, j)

                        for anchor_idx, anchor in enumerate(tile_anchors):
                            iou = box_iou(bbox, anchor)
                            if iou > iou_threshold:
                                associations.append(((i, j), anchor_idx, iou))
                associations = sorted(associations, key=lambda x: x[2], reverse=True)

    def _generate_tile_anchors(self, i: int, j: int):
        im_w = self.ground_truth_generation_settings.input_image_width
        im_h = self.ground_truth_generation_settings.input_image_height
        rows = self.ground_truth_generation_settings.grid_rows
        cols = self.ground_truth_generation_settings.grid_cols

        tile_center_x = im_w // rows
        tile_center_y = im_h // cols

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

        return anchors


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
    TRAIN_DATA_PATH = Path(
        "C:\\Projects\\Learning\\ObjectDetection_PascalVoc2012\\Datasets\\toy_set"
    )
    TRAIN_ANNOTATIONS_PATH = Path(
        "C:\\Projects\\Learning\\ObjectDetection_PascalVoc2012\\Datasets\\train_val\\Annotations"
    )

    TEST_DATA_PATH = Path(
        "C:\\Projects\\Learning\\ObjectDetection_PascalVoc2012\\Data\\test\\JPEGImages"
    )

    # We transform them to Tensors of normalized range [-1, 1]
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5), (0.5, 0.5))]
    # )
    # dataset_path: Path, annotations_path: Path, dataset_name: str, transform
    train_dataset = PascalVocDataset(
        TRAIN_DATA_PATH, TRAIN_ANNOTATIONS_PATH, "train_val", None
    )
    # test_dataset = PascalVocDataset(TEST_DATA_PATH, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for i, sample in enumerate(train_dataset):
        img, annotations = sample

        show_image_with_annotations(img, annotations)
        if i == 3:
            break
