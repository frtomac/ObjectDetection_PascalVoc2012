from lxml import etree
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List, Union

from Annotation import Annotation, BoundingBox, parse_xml_image_annotations


def show_image(image_path: Path):
    if not image_path.exists():
        exit(f"Image not found: {image_path}. Aborting.")

    image = Image.open(image_path)
    image.show()


def show_image_and_annotations(image_path: Path, annotations: List[Annotation]):
    if not image_path.exists():
        exit(f"Image not found: {image_path}. Aborting.")

    image = Image.open(image_path)

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


# def _add_bbox_to_image_plot(image, bbox: Union[BoundingBox, List[int, int, int, int]]):
#     """TBA."""

#     im_name = root.find("filename").text

#     # Load the image
#     image_path = PROJECT_DIR / f"Datasets/train_val/JPEGImages/{im_name}"
#     image = Image.open(image_path)

#     # Create a figure and axis
#     fig, ax = plt.subplots()

#     # Display the image
#     ax.imshow(image)

#     # Define the bounding box coordinates (x, y, width, height)
#     bbox = [50, 50, 100, 150]  # Example coordinates

#     # Create a rectangle patch
#     rect = patches.Rectangle(
#         (bbox[0], bbox[1]),
#         bbox[2],
#         bbox[3],
#         linewidth=2,
#         edgecolor="r",
#         facecolor="none",
#     )

#     # Add the rectangle to the plot
#     ax.add_patch(rect)

#     # Show the plot
#     plt.show()


if __name__ == "__main__":
    PROJECT_NAME = "ObjectDetection_PascalVoc2012"
    PROJECT_DIR = Path.cwd() / PROJECT_NAME
    annotation_path = PROJECT_DIR / "Datasets/train_val/Annotations/2007_000762.xml"

    # parsed_dict = untangle.parse(str(annotation_path))

    root = etree.parse(annotation_path).getroot()
    im_name = root.find("filename").text

    image_path = PROJECT_DIR / f"Datasets/train_val/JPEGImages/{im_name}"
    annotations = parse_xml_image_annotations(annotation_path)
    show_image_and_annotations(image_path, annotations)
