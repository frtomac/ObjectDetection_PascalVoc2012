from PIL import Image
from pathlib import Path
from typing import Any
from torch import Tensor
from torchvision.transforms import ToPILImage

from Lib.Exceptions import ImageOrAnnotationTypeNotSupportedException


def save_image(img: Any, output_dir: Path, img_name: str) -> None:
    # TODO: check that image name contains extension; handle if necessary.
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(
                f"DataIO.save_image: You don't have the permission to create {output_dir}"
            )
            exit()
        except OSError as e:
            print(
                f"DataIO.save_image: An OS error occurred when trying to create directory {output_dir}: {e}."
            )
            exit()

    if isinstance(img, Image.Image):
        img.save(output_dir / img_name)
    elif isinstance(img, Tensor):
        img = ToPILImage()(img)
        # img.permute(1, 2, 0)
        img.save(output_dir / img_name)
    else:
        raise ImageOrAnnotationTypeNotSupportedException(
            f"DataIO.save_image: supported image types are PIL Image and torch Tensor but received {type(img)}."
            "If another type is needed, update the function."
        )
