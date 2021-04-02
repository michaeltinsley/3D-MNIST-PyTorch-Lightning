"""
Simple 3D CNN model.
"""
from typing import Tuple

from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

class MNIST3DCNN(LightningModule):
    """
    A simple 3D CNN classification model.
    """
    def __init__(self, num_classes: int = 10, input_dimension: Tuple[int, int, int] = (16, 16, 16)) -> None:
        """

        """
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.input_dimension = input_dimension

        self.init_model()

    def init_model(self) -> None:
        """
        Initialise the layers of the model.
        """
        self.conv_1 = torch.nn.Conv3d(
            in_channels=self.input_dimension[0],
            out_channels=self.input_dimension[0] * 2,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=0,
            padding_mode='zeros',
            dilation=1,
            groups=1,
            bias=True,
        )

        self.dropout_1 = torch.nn.Dropout3d(p=0.5, inplace=False)

        self.conv_2 = torch.nn.Conv3d(
            in_channels=self.conv_1.out_channels,
            out_channels=self.conv_1.out_channels * 2,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=0,
            padding_mode='zeros',
            dilation=1,
            groups=1,
            bias=True,
        )

        self.dropout_2 = torch.nn.Dropout3d(p=0.5, inplace=False)

        self.flatten = torch.nn.Flatten()

        self.fc_1 = torch.nn.Linear(
            in_features=self.flatten.,
            out_features=
        )

    def forward(self, *args, **kwargs):
        """
        Forward defines the prediction/inference actions
        """




