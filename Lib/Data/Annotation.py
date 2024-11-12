from dataclasses import dataclass
from enum import auto
from lxml import etree
from pathlib import Path
from typing import List, TypeAlias
from strenum import StrEnum


class InvalidBoundingBoxException(Exception):
    """Custom exception thrown when the bounding box parsed from annotations is invalid.

    Bounding box is considered VALID if 0 <= x_min < x_max <= image_width and 0 <= y_min < y_max < image_height.
    """

    pass


class InvalidCategoryException(Exception):
    """Custom exception thrown when the category parsed from annotations is invalid.

    For a list of valid categories, see the 'Category' enumeration in Lib/Data/Annotation.py.
    """

    pass


class Category(StrEnum):
    aeroplane = auto()
    bicycle = auto()
    bird = auto()
    boat = auto()
    bottle = auto()
    bus = auto()
    car = auto()
    cat = auto()
    chair = auto()
    cow = auto()
    diningtable = auto()
    dog = auto()
    horse = auto()
    motorbike = auto()
    person = auto()
    pottedplant = auto()
    sheep = auto()
    sofa = auto()
    train = auto()
    tvmonitor = auto()

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


@dataclass
class ImageProperties:
    name: str
    width: int
    height: int
    depth: int


@dataclass
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int = None
    height: int = None

    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min

    def __str__(self):
        return f"(x_min, y_min) = ({self.x_min}, {self.y_min}), (x_max, y_max) = ({self.x_max}, {self.y_max})."

    def _bbox_is_valid(self, im_w, im_h) -> bool:
        """Verify bounding box

        Return:
            - True if 0 <= x_min < x_max <= image_width and 0 <= y_min < y_max <= image_height
            - False otherwise.
        """
        return (
            0 <= self.x_min < self.x_max <= im_w
            and 0 <= self.y_min < self.y_max <= im_h
        )


@dataclass
class Annotation:
    category: Category
    bbox: BoundingBox
    image_properties: ImageProperties

    def __init__(
        self, category: Category, bbox: BoundingBox, image_properties: ImageProperties
    ):
        self.category = category
        self.bbox = bbox
        self.image_properties = image_properties

        if not Category.has_value(category):
            raise InvalidCategoryException(
                f"Annotations for file {self.image_properties.name} contain an invalid category: {category}."
            )

        if not self.bbox._bbox_is_valid(
            self.image_properties.width, self.image_properties.height
        ):
            raise InvalidBoundingBoxException(
                f"Annotations for file {self.image_properties.name} contain an invalid bounding box: {self.bbox}."
            )

    def _transfrom_annotation(self, transform):  # TODO: add type.

        pass


ImageAnnotations: TypeAlias = List[Annotation]


@dataclass
class Detection:
    category: Category
    bbox: BoundingBox
    existence_probability: float

    def __init__(
        self, category: Category, bbox: BoundingBox, existance_prob: float = 0.0
    ):
        self.category = category
        self.bbox = bbox
        self.existence_probability = existance_prob

        if not Category.has_value(category):
            raise InvalidCategoryException(
                f"Invalid category for detection: {category}."
            )

        # TODO: Add checks for bbox? How to pass image dimensions?


ImageDetections: TypeAlias = List[Detection]


def parse_xml_image_annotations(annotation_path: Path) -> ImageAnnotations:
    if not annotation_path.exists():
        raise FileNotFoundError

    annotations = []
    root = etree.parse(annotation_path).getroot()

    im_name = root.find("filename").text
    im_w = int(root.find("size/width").text)
    im_h = int(root.find("size/height").text)
    im_d = int(root.find("size/depth").text)
    image_properties = ImageProperties(im_name, im_w, im_h, im_d)

    for obj in root.findall("object"):
        category = obj.find("name").text

        bbox = BoundingBox(
            x_min=int(obj.find("bndbox/xmin").text),
            y_min=int(obj.find("bndbox/ymin").text),
            x_max=int(obj.find("bndbox/xmax").text),
            y_max=int(obj.find("bndbox/ymax").text),
        )

        try:
            annotations.append(Annotation(category, bbox, image_properties))
        except (InvalidBoundingBoxException, InvalidCategoryException) as e:
            print(e)
            continue

    return annotations


if __name__ == "__main__":
    PROJECT_NAME = "ObjectDetection_PascalVoc2012"
    PROJECT_DIR = Path.cwd() / PROJECT_NAME

    annotation_path = PROJECT_DIR / "Datasets/train_val/Annotations/2007_000423.xml"
    annotations = parse_xml_image_annotations(annotation_path)

    print("Stop")
