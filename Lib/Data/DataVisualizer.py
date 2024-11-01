from lxml import etree
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import Any, List
from torch import Tensor
from torchvision.transforms import ToPILImage

from Annotation import (
    Annotation,
    BoundingBox,
    parse_xml_image_annotations,
    ImageAnnotations,
)


COLOR = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Yellow": (255, 255, 0),
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Orange": (255, 165, 0),
    "Purple": (128, 0, 128),
}
color_values = list(COLOR.values())
color_values = [(val[0] / 255, val[1] / 255, val[2] / 255) for val in color_values]


def show_image_on_path(image_path: Path):
    if not image_path.exists():
        exit(f"Image not found: {image_path}. Aborting.")

    image = Image.open(image_path)
    image.show()


def show_image_on_path_with_annotations(
    image_path: Path, annotations: ImageAnnotations
):
    if not image_path.exists():
        exit(f"Image not found: {image_path}. Aborting.")

    image = Image.open(image_path)

    show_image_with_annotations(image, annotations)


def show_image_with_annotations(image: Any, annotations: ImageAnnotations):

    if isinstance(image, Tensor):
        image = ToPILImage()(image)

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image)

    for ann in annotations:
        rect = patches.Rectangle(
            (ann.bbox.x_min, ann.bbox.y_min),
            ann.bbox.width,
            ann.bbox.height,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        # Add the rectangle to the plot
        ax.add_patch(rect)

        ax.text(
            ann.bbox.x_min,
            ann.bbox.y_min - 10,
            ann.category,
            color="white",
            fontsize=12,
            bbox=dict(facecolor="red", alpha=0.5),
        )

    plt.show()


def show_annotations_on_image_with_grid_and_associated_anchors(
    img, anchor_annotation_pairs, grid_cols, grid_rows
):
    if isinstance(image, Tensor):
        image = ToPILImage()(image)

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)

    im_w, im_h = (
        img.size
    )  # TODO: works for PIL.Image objects - what to do with other possibilities?

    # Add grid on image
    ax.grid(which="major", axis="both", linestyle="-", color=COLOR["Black"])
    ax.set_xticks(range(0, im_w, im_w // grid_cols))  # Adjust the step size as needed
    ax.set_yticks(range(0, im_h, im_h // grid_rows))  # Adjust the step size as needed¸

    img_annotations = [
        anchor_annotation_pair[1]
        for anchor_annotation_pair in anchor_annotation_pairs.values()
    ]

    # show_image_with_annotations(img, img_annotations)
    for i, ann in enumerate(img_annotations):
        edgecolor = color_values[i % len(color_values)]

        bbox = BoundingBox(ann[0][0], ann[0][1], ann[0][2], ann[0][3])
        rect = patches.Rectangle(
            (bbox.x_min, bbox.y_min),
            bbox.width,
            bbox.height,
            linewidth=2,
            edgecolor=edgecolor,
            facecolor="none",
        )
        # Add the rectangle to the plot
        ax.add_patch(rect)

        anchor = list(anchor_annotation_pairs.values())[i][0]
        anchor_bbox = BoundingBox(anchor[0], anchor[1], anchor[2], anchor[3])
        anchor_rect = patches.Rectangle(
            (anchor_bbox.x_min, anchor_bbox.y_min),
            anchor_bbox.width,
            anchor_bbox.height,
            linewidth=2,
            edgecolor=edgecolor,
            facecolor="none",
        )
        # Add the rectangle to the plot
        ax.add_patch(anchor_rect)

        ax.text(
            bbox.x_min,
            bbox.y_min - 10,
            ann[1],
            color="white",
            fontsize=12,
            bbox=dict(facecolor="red", alpha=0.5),
        )

    plt.show()


def show_grid_and_anchors_on_image(img, anchors, grid_cols=3, grid_rows=3):

    if isinstance(image, Tensor):
        image = ToPILImage()(image)
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    im_w, im_h = (
        img.size
    )  # TODO: works for PIL.Image objects - what to do with other possibilities?

    # Add grid on image
    ax.grid(which="major", axis="both", linestyle="-", color=COLOR["Black"])
    ax.set_xticks(range(0, im_w, im_w // grid_cols))  # Adjust the step size as needed
    ax.set_yticks(range(0, im_h, im_w // grid_rows))  # Adjust the step size as needed¸

    for anchor in anchors:

        bbox = BoundingBox(anchor[0], anchor[1], anchor[2], anchor[3])
        rect = patches.Rectangle(
            (bbox.x_min, bbox.y_min),
            bbox.width,
            bbox.height,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        # Add the rectangle to the plot
        ax.add_patch(rect)

    plt.show()


if __name__ == "__main__":
    PROJECT_DIR = Path("C:\\Projects\\ObjectDetection_PascalVoc2012")
    annotation_path = PROJECT_DIR / "Datasets\\train_val\\Annotations\\2007_000762.xml"

    # parsed_dict = untangle.parse(str(annotation_path))

    root = etree.parse(annotation_path).getroot()
    im_name = root.find("filename").text

    image_path = PROJECT_DIR / f"Datasets/train_val/JPEGImages/{im_name}"
    annotations = parse_xml_image_annotations(annotation_path)
    show_image_on_path_with_annotations(image_path, annotations)
