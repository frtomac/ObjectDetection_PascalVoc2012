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
        self, in_channels=3, grid_rows=3, grid_cols=3, num_classes=20, num_anchors=3
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.existence_prob = (
            nn.Sequential(  # output size: grid_rows * grid_cols * num_anchors
                nn.Linear(
                    in_features=1 * 64 * 43 * 43,
                    out_features=grid_rows * grid_cols * num_anchors,
                ),
                nn.Sigmoid(),
            )
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(
                in_features=1 * 64 * 43 * 43,
                out_features=4 * grid_rows * grid_cols * num_anchors,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=1 * 64 * 43 * 43,
                out_features=num_classes * grid_rows * grid_cols * num_anchors,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)

        # x = x.view(-1, 64 * 48 * 48)  # Flatten the tensor
        # x = x.view(-1, 1 * 64 * 43 * 43)  # Flatten the tensor
        x = x.view(x.size(0), -1)

        existence_prob = self.existence_prob(x)
        bbox_offsets = self.bbox_regressor(x)
        class_confidences = self.classifier(x)

        return existence_prob, bbox_offsets, class_confidences


if __name__ == "__main__":

    # Summaries
    batch_size = 1
    input_image_width = 180
    input_image_height = 180
    print(f"device = {device}")

    detector = Detector().to(device)
    summary(
        detector,
        input_size=(3, input_image_width, input_image_height),
        batch_size=batch_size,
    )
