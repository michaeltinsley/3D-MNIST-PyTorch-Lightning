"""
3D MNIST runner.
"""
from configparser import ConfigParser

from comet_ml import Experiment  # noqa # pylint: disable=unused-import
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger

from mnist_3d_pytorch_lightning import MNIST3DCNN, MNIST3DDataModule

if __name__ == "__main__":

    config = ConfigParser()
    config.read("config.ini")
    comet_config = config["COMET_LOGGING"]

    comet_logger = CometLogger(
        api_key=comet_config.get("api_key"),
        workspace=comet_config.get("workspace"),
        project_name=comet_config.get("project_name"),
        experiment_name=comet_config.get("experiment_name"),
    )

    model = MNIST3DCNN()
    data = MNIST3DDataModule()

    trainer = Trainer(logger=comet_logger)
    trainer.fit(model, data)
