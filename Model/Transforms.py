import abc
from dataclasses import dataclass, field
from math import log
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Type, List

from torchvision.ops import box_iou
import torch
import torchvision.transforms as transforms
import numpy as np
import numpy.typing
import pandas as pd

from Lib.Data.Annotation import Annotation, BoundingBox


class ImageOrAnnotationTypeNotSupportedException(Exception):
    """Custom exception thrown when image or annotation type is not supported."""

    pass


class TransformWrapper(abc.ABC):
    # TODO: fix type hinting in the entire class.
    transform: Callable[
        ..., torch.Tensor
    ]  # TODO: think about making this a mandatory property
    "This means: take in any number of arguments of any type, but return torch.Tensor"

    def __call__(self, img):
        if isinstance(img, Annotation):
            return self.transform_annotation(img)
        elif isinstance(img, Image.Image):
            return self.transform(img)
        raise ImageOrAnnotationTypeNotSupportedException

    @abc.abstractmethod
    def transform_annotation(self, annotations: Annotation) -> Any:
        raise NotImplementedError


class Resize(TransformWrapper):
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w
        self.transform = transforms.Resize(size=(new_h, new_w))

    # def __call__(self, img):
    #     return super.__call__(img)

    def transform_annotation(self, annotation: Annotation) -> Annotation:
        im_w = annotation.image_properties.width
        im_h = annotation.image_properties.height

        width_ratio = self.new_w / im_w
        height_ratio = self.new_h / im_h

        x_min = annotation.bbox.x_min * width_ratio
        y_min = annotation.bbox.y_min * height_ratio
        x_max = annotation.bbox.x_max * width_ratio
        y_max = annotation.bbox.y_max * height_ratio

        return Annotation(
            annotation.category,
            BoundingBox(x_min, y_min, x_max, y_max),
            image_properties=annotation.image_properties,
        )


class ToTensor(TransformWrapper):
    def transform_annotation(self, annotation: Annotation):
        return annotation
