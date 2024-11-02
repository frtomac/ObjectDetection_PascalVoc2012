import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
import torchvision.transforms as transforms
from torchvision.ops import ciou_loss  # complete_box_iou_loss
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
    learning_rate = 0.08

    model = Detector().to(device)
    summary(model, input_size=(3, input_img_h, input_img_w), batch_size=batch_size)

    criterion_existence_prob = nn.CrossEntropyLoss()
    criterion_bbox_regression = nn.MSELoss()
    criterion_classes = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    n_total_steps = len(train_dataloader)
    for epoch in range(num_epochs):

        running_loss = 0.0

        for i, (batch_images, batch_ground_truths) in enumerate(train_dataloader):
            batch_images = batch_images.to(torch.float32).to(device)

            # Forward pass
            (
                batch_output_existence,
                batch_output_bbox_offsets,
                batch_output_classes,
            ) = model(batch_images)
            batch_output_existence = batch_output_existence.to(torch.float32)
            batch_output_bbox_offsets = batch_output_bbox_offsets.to(torch.float32)
            batch_output_classes = batch_output_classes.to(torch.float32)

            """
            ground_truths = {
            "existence_probs": [],
            "bbox_offsets": [],
            "class_confidences": [],
            """

            batch_gt_existence_probs = [
                ex_probs.to(torch.float32).to(device)
                for ex_probs in batch_ground_truths["existence_probs"]
            ]
            batch_gt_existence_probs = torch.stack(batch_gt_existence_probs, dim=1)

            batch_gt_bbox_offsets = [
                offsets.to(torch.float32).to(device)
                for offsets in batch_ground_truths["bbox_offsets"]
            ]
            batch_gt_bbox_offsets = torch.stack(batch_gt_bbox_offsets, dim=1)

            batch_gt_class_confidences = [
                conf.to(torch.float32).to(device)
                for conf in batch_ground_truths["class_confidences"]
            ]
            batch_gt_class_confidences = torch.stack(batch_gt_class_confidences, dim=1)

            loss_existence_prob = criterion_existence_prob(
                batch_output_existence, batch_gt_existence_probs
            )
            loss_bbox_regression = criterion_bbox_regression(
                batch_output_bbox_offsets, batch_gt_bbox_offsets
            )
            loss_classes = criterion_classes(
                batch_output_classes, batch_gt_class_confidences
            )

            total_loss = (loss_existence_prob + loss_bbox_regression + loss_classes).to(
                torch.float32
            )
            # Backward and optimize
            # total_loss = total_loss.to(torch.float32)<
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += total_loss.item()

        print(f"[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}")

    print("Finished Training")
    PATH = Path("cnn.pth")
    torch.save(model.state_dict(), PATH)


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
