"""
Simple 3D CNN model.
"""
from typing import Tuple
from .data_module import MNIST3DDataModule
from pytorch_lightning import LightningModule
import torch


class MNIST3DCNN(LightningModule):
    """
    A simple 3D CNN classification PyTorch Lightning module.
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_dimension: Tuple[int, int, int] = (1, 16, 16, 16),
    ) -> None:
        """
        A simple 3D CNN classification PyTorch Lightning module.

        :param num_classes: Number of output classes (default is 10)
        :param input_dimension: Dimension of input tensor (default is (1, 16, 16, 16))
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.input_dimension = input_dimension

        self.model = self.init_model()
        self._example_input_array = torch.randn((1, *self.input_dimension))

    def conv_block(self, in_channels: int, out_channels: int) -> torch.nn.Sequential:
        """
        Builds a repeatable convolutional block.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :return: The sequential convolutional block
        """
        return torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3, 3),
                bias=True,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

    def init_model(self) -> torch.nn.Sequential:
        """
        Initialise the layers of the model.

        :return: The PyTorch sequential model
        """
        model = torch.nn.Sequential(
            self.conv_block(1, 32),
            self.conv_block(32, 64),
            # FC block
            torch.nn.Flatten(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(512, 10),
        )

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward defines the prediction/inference actions.

        :param x: The input tensor
        :return: The model output tensor
        """
        output = self.model(x)
        return torch.nn.functional.log_softmax(output, dim=-1)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        """
        Returns the training loss and logs loss.

        :param batch: The input and target training data
        :param batch_idx: The index of the given batch
        :return: The training loss
        """
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Logs the validation loss.

        :param batch: The input and target validation data
        :param batch_idx: The index of the given batch
        """
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Logs the test loss.

        :param batch: The input and target test data
        :param batch_idx: The index of the given batch
        """
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures optimizers for the model.

        :return: The configured optimisers
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)
