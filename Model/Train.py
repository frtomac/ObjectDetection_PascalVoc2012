import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.ops import complete_box_iou_loss
from pathlib import Path

from Model.Model import Detector, device
from Model.Dataset import PascalVocDataset
from Lib import Data

# TODO: move what you can to Config.py


def run_model(
    train: bool = True, eval: bool = True, generate_ground_truth: bool = True
):
    if generate_ground_truth:
        # GroundTruthGenerator
        pass
    pass


def train(
    train_data_path, train_annotations_path, test_data_path, test_annotations_path
):
    # We transform them to Tensors of normalized range [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )
    train_dataset = PascalVocDataset(
        train_data_path, train_annotations_path, "train", transform
    )
    test_dataset = PascalVocDataset(
        test_data_path, test_annotations_path, "test", transform
    )

    num_epochs = 1
    batch_size = 2
    learning_rate = 0.001

    model = Detector().to(device)
    # summary(model, input_size=(1, 28, 28), batch_size=batch_size)

    criterion_existence_prob = nn.CrossEntropyLoss()
    criterion_bbox_regression = complete_box_iou_loss()
    criterion_classes = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    n_total_steps = len(train_dataloader)
    for epoch in range(num_epochs):

        running_loss = 0.0

        for i, (images, annotations) in enumerate(train_dataloader):
            images = images.to(device)
            annotations = annotations.to(device)

            # TODO: Finish this part of the code: loss calculation etc.

            # Forward pass
            outputs = model(images)
            loss_existence_prob = criterion_existence_prob(outputs, annotations)

            # # Backward and optimize
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # running_loss += loss.item()

        print(f"[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}")

    print("Finished Training")
    PATH = Path("cnn.pth")
    torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    TRAIN_DATA_PATH = Path(
        "C:\\Projects\\Learning\\ObjectDetection_PascalVoc2012\\Datasets\\toy_set"
    )
    TRAIN_ANNOTATIONS_PATH = Path(
        "C:\\Projects\\Learning\\ObjectDetection_PascalVoc2012\\Datasets\\train_val\\Annotations"
    )

    TEST_DATA_PATH = Path(
        "C:\\Projects\\Learning\\ObjectDetection_PascalVoc2012\\Datasets\\test\\JPEGImages"
    )
    TEST_ANNOTATIONS_PATH = Path(
        "C:\\Projects\\Learning\\ObjectDetection_PascalVoc2012\\Datasets\\test\\Annotations"
    )

    train(
        TRAIN_DATA_PATH, TRAIN_ANNOTATIONS_PATH, TEST_DATA_PATH, TEST_ANNOTATIONS_PATH
    )
