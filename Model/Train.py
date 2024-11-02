import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
import torchvision.transforms as transforms
from torchvision.ops import complete_box_iou_loss
from pathlib import Path
from typing import List, Tuple

import Transforms
from Model import Detector, device
from Dataset import PascalVocDataset, GroundTruthGenerationSettings
from Lib import Data

# TODO: move what you can to Config.py


# def run_model(
#     train: bool = True, eval: bool = True, generate_ground_truth: bool = True
# ):
#     if generate_ground_truth:
#         # GroundTruthGenerator
#         pass
#     pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    train_images_path,
    train_annotations_path,
    gt_path,
    num_classes=20,
    grid_cols=3,
    grid_rows=3,
    num_anchors=3,
    input_img_h=180,
    input_img_w=180,
):
    # We transform them to Tensors of normalized range [-1, 1]
    transform = transforms.Compose(
        [
            Transforms.ToTensor(),
            Transforms.Resize(new_h=input_img_h, new_w=input_img_w),
            Transforms.Normalize(),
        ]
    )
    gt_generation_settings = GroundTruthGenerationSettings(ground_truth_path=gt_path)
    train_dataset = PascalVocDataset(
        train_images_path,
        train_annotations_path,
        gt_generation_settings,
        "train_val_toy",
        transform,
    )
    # test_dataset = PascalVocDataset(
    #     test_images_path, test_annotations_path, "test", transform
    # )

    num_epochs = 1
    batch_size = 3
    learning_rate = 0.001

    model = Detector().to(device)
    summary(model, input_size=(3, input_img_h, input_img_w), batch_size=batch_size)

    criterion_existence_prob = nn.CrossEntropyLoss()
    criterion_bbox_regression = complete_box_iou_loss()
    criterion_classes = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    n_total_steps = len(train_dataloader)
    for epoch in range(num_epochs):

        running_loss = 0.0

        for i, (batch_images, batch_ground_truths) in enumerate(train_dataloader):
            batch_images = batch_images.to(device)
            batch_ground_truths = batch_ground_truths.to(device)

            # Forward pass
            batch_outputs = model(batch_images)

            output_existence, output_bbox_regression, output_classes = (
                _unpack_batch_model_output(
                    batch_outputs, num_classes, grid_cols, grid_rows, num_anchors
                )
            )
            # annotations_existence = ...
            # annotations_bbox_regression = ...
            # annotations_classes = ...
            # loss_existence_prob = criterion_existence_prob(
            #     output_existence, annotations_existence
            # )
            # loss_bbox_regression = criterion_bbox_regression(
            #     output_bbox_regression, annotations_bbox_regression
            # )
            # loss_classes = criterion_classes(output_classes, annotations_classes)

            # # Backward and optimize
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # running_loss += loss.item()

        # print(f"[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}")

    print("Finished Training")
    PATH = Path("cnn.pth")
    torch.save(model.state_dict(), PATH)


def _unpack_batch_model_output(
    outputs: List, num_classes, grid_cols, grid_rows, num_anchors
):
    existence_probs = []
    offsets = []
    class_confs = []
    for batch_output in outputs:  # TODO: check this!
        one_batch_probs, one_batch_offsets, one_batch_confs = _unpack_model_output(
            batch_output, num_classes, grid_cols, grid_rows, num_anchors
        )

        existence_probs.append(one_batch_probs)
        offsets.append(one_batch_offsets)
        class_confs.append(one_batch_confs)

    return existence_probs, offsets, class_confs


def _unpack_model_output(
    output: List, num_classes, grid_cols, grid_rows, num_anchors
) -> Tuple[List, List, List]:
    """Unpacks the model output vector and returns existence probabilities, box offsets and class confidences."""

    assert len(output) == (1 + 4 + num_classes) * grid_rows * grid_cols * num_anchors

    one_anchor_gt_len = 1 + 4 + num_classes

    existence_probs = [
        output[i * one_anchor_gt_len]
        for i in range(0, grid_rows * grid_cols * num_anchors)
    ]
    box_offsets = []
    for i in range(0, grid_rows * grid_cols * num_anchors):
        box_offsets.extend(
            output[1 + i * one_anchor_gt_len : 5 + i * one_anchor_gt_len]
        )
    class_confs = []
    for i in range(0, grid_rows * grid_cols * num_anchors):
        class_confs.extend(
            output[5 + i * one_anchor_gt_len : (i + 1) * one_anchor_gt_len]
        )

    return existence_probs, box_offsets, class_confs


if __name__ == "__main__":
    TRAIN_IMAGES_PATH = Path(
        "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\train_val_toy\\JPEGImages"
    )
    TRAIN_ANNOTATIONS_PATH = Path(
        "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\train_val_toy\\Annotations"
    )
    TRAIN_GT_PATH = Path(
        "C:\\Projects\\ObjectDetection_PascalVoc2012\\Datasets\\train_val_toy\\GroundTruth"
    )

    TEST_DATA_PATH = Path(
        "C:\\Projects\\Learning\\ObjectDetection_PascalVoc2012\\Datasets\\test\\JPEGImages"
    )
    TEST_ANNOTATIONS_PATH = Path(
        "C:\\Projects\\Learning\\ObjectDetection_PascalVoc2012\\Datasets\\test\\Annotations"
    )

    train(TRAIN_IMAGES_PATH, TRAIN_ANNOTATIONS_PATH, TRAIN_GT_PATH)
