import torch

from torch import nn

#Now, we finally work on the model. We will focus on recreating the model from TinyVGG
class VGGImposter(nn.Module):
    """
    A simplified VGG-like architecture with two convolutional blocks followed by
    a fully connected layer for classification.

    The architecture includes:
    - Two convolutional blocks, each consisting of two convolutional layers
      followed by ReLU activations and a max pooling layer.
    - A final classifier consisting of a flattening layer and a linear layer
      to produce class scores.

    Args:
        input_channels (int): Number of input channels (e.g., 3 for RGB images).
        hidden_units (int): Number of filters in the convolutional layers.
        output_classes (int): Number of output classes for classification.

    Example:
        model = TinyVGG(input_channels=3, hidden_units=32, output_classes=10)
    """

    def __init__(self, input_channels: int, hidden_units: int, output_classes: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 18 * 18,  # Assuming input images are 64x64
                      out_features=output_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))