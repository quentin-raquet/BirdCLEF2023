import os
import argparse
import yaml
import torch
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from src.data import BirdDataModule
from src.model import BirdModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="test",
    )
    parser.add_argument(
        "--db_host",
        default=None,
    )
    parser.add_argument(
        "--db_token",
        default=None,
    )
    args = parser.parse_args()
    return args


def main():
    inputs = parse_args()
    env = inputs.env

    config_path = Path("config/config.yaml")

    with config_path.open() as fh:
        config = yaml.safe_load(fh)

    mlf_logger = None
    if env == "databricks":
        import mlflow

        os.environ["DATABRICKS_HOST"] = inputs.db_host
        os.environ["DATABRICKS_TOKEN"] = inputs.db_token
        mlf_logger = MLFlowLogger(
            experiment_name=config[env]["trainer"]["mlflow_experiment_name"],
            tracking_uri="databricks",
            log_model=True,
        )

    seed_everything(1992, workers=True)
    torch.set_float32_matmul_precision("medium")
    datamodule = BirdDataModule(**config[env]["datamodule"])
    datamodule.setup()

    model = BirdModel(**config[env]["model"])

    trainer_cfg = config[env]["trainer"]

    trainer = Trainer(
        accelerator=trainer_cfg["device_type"],
        devices=trainer_cfg["devices"],
        max_epochs=trainer_cfg["max_epochs"],
        callbacks=[
            ModelCheckpoint(
                monitor=trainer_cfg["early_stopping_metric"],
                mode="max",
                save_weights_only=True,
                filename="{epoch}-{train_map:.4f}-{val_map:.4f}",
            ),
            EarlyStopping(
                monitor=trainer_cfg["early_stopping_metric"],
                mode="max",
                min_delta=0.0,
                patience=trainer_cfg["patience"],
            ),
        ],
        logger=mlf_logger,
        default_root_dir=trainer_cfg["default_dir"],
    )
    trainer.fit(model, datamodule)
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    if env == "databricks":
        experiment = mlflow.get_experiment_by_name(
            config[env]["trainer"]["mlflow_experiment_name"]
        )
        exp_id = experiment.experiment_id
        with mlflow.start_run(experiment_id=exp_id, run_id=mlf_logger.run_id) as run:
            mlflow.pytorch.log_model(model.backbone, "model")


if __name__ == "__main__":
    main()
