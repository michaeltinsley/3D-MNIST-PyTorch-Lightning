"""
PyTorch Lightning DataModule for 3D MNIST.
"""
from typing import Optional, Tuple

import h5py
import torch
from kaggle.api import KaggleApi
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class MNIST3DDataset(Dataset):
    """
    PyTorch Dataset wrapper for 3D MNIST.
    """

    def __init__(
        self,
        dataset_path: str,
        train: bool = True,
        output_shape: Tuple[int, int, int] = (1, 16, 16, 16),
    ) -> None:
        """
        Parses the 3D MNIST dataset into a PyTorch Dataset object.

        :param dataset_path: The filepath to the full dataset file
        :param train: Create training dataset if True, test dataset if False
                      (default is True)
        :param output_shape: The shape of the output data (default is (16, 16, 16))
        """
        self.dataset_path = dataset_path
        self.train = train
        self.output_shape = output_shape

        self.data, self.targets = self.get_dataset()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Supports fetching a data sample for a given key.

        :param index: The index
        :return: The indexed datum and corresponding label
        """
        voxel, target = self.data[index], int(self.targets[index])
        return (
            torch.reshape(voxel, self.output_shape),  # pylint: disable=no-member
            target,
        )

    def __len__(self) -> int:
        """
        Supports fetching the length of the dataset

        :return: The length of the dataset
        """
        return len(self.data)

    def get_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts and returns the specified dataset.

        :return: The specified dataset, data and targets
        """
        with h5py.File(self.dataset_path, "r") as h5_file:
            x_data = h5_file[f"X_{'train' if self.train else 'test'}"][:]
            y_data = h5_file[f"y_{'train' if self.train else 'test'}"][:]

        return (
            torch.tensor(x_data).float(),  # pylint: disable=not-callables
            torch.tensor(y_data),
        )


class MNIST3DDataModule(LightningDataModule):
    """
    3D MNIST PyTorch Lightning DataModule.
    """

    # pylint: disable=too-many-instance-attributes
    KAGGLE_DATASET_OWNER_SLUG = "daavoo"
    KAGGLE_DATASET_SLUG = "3d-mnist"
    KAGGLE_DATASET_DATASET_FILE = "full_dataset_vectors.h5"

    def __init__(
        self,
        data_dir: Optional[str] = "./data",
        batch_size: Optional[int] = 32,
        validation_split: Optional[float] = 0.1,
        quiet: Optional[bool] = False,
    ) -> None:
        """
        A 3D MNIST data module object.

        :param data_dir: The directory to download the dataset to (default is `./data`)
        :param batch_size: The batch size (default is `32`)
        :param validation_split: Amount of training set to use for validation, must be
                                 between 0 and 1 (default is 0.1)
        :param quiet: Suppress download progress output (default is False)
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.quiet = quiet

        self.dataset_path = f"{self.data_dir}/{self.KAGGLE_DATASET_DATASET_FILE}"
        self.dims = (16, 16, 16)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def download_data(self) -> None:
        """
        Downloads the 3D MNIST dataset.
        """
        api_client = KaggleApi()
        api_client.authenticate()
        api_client.dataset_download_files(
            dataset=f"{self.KAGGLE_DATASET_OWNER_SLUG}/{self.KAGGLE_DATASET_SLUG}",
            path=self.data_dir,
            quiet=self.quiet,
            unzip=True,
        )

    def prepare_data(self, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        """
        Download the dataset.
        """
        self.download_data()

    def setup(
        self, stage: Optional[str] = None  # pylint: disable=unused-argument
    ) -> None:
        """
        Load and prepare the dataset.

        :param stage: Unused stage var
        """
        train_dataset = MNIST3DDataset(dataset_path=self.dataset_path, train=True)
        val_split = round(len(train_dataset) * self.validation_split)
        train_split = len(train_dataset) - val_split
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_split, val_split]
        )
        self.test_dataset = MNIST3DDataset(dataset_path=self.dataset_path, train=False)

    def train_dataloader(self) -> DataLoader:
        """
        The training dataset.

        :return: The training DataLoader object
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """
        The validation dataset.

        :return: The validation DataLoader object
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """
        The test dataset.

        :return: The test DataLoader object
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
