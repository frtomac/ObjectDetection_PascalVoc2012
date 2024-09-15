import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from torchsummary import summary

# TODO: move what you can to Config.py

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector(nn.Module):
    def __init__(
        self, in_channels=3, grid_rows=9, grid_cols=9, num_classes=20, num_anchors=3
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(
            in_features=64 * 48 * 48,
            out_features=grid_rows * grid_cols * num_classes * num_anchors,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = x.view(-1, 64 * 48 * 48)  # Flatten the tensor
        x = self.fc1(x)

        return x


if __name__ == "__main__":

    # Summaries
    batch_size = 1
    input_image_width = 200
    input_image_height = 200
    print(f"device = {device}")

    backbone = Detector().to(device)
    summary(
        backbone,
        input_size=(3, input_image_width, input_image_height),
        batch_size=batch_size,
    )
