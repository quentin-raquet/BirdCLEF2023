import argparse
import yaml
import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from config.labels import LABELS
from src.model import BirdModel
from src.data import PredictDataset, predict_collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="test",
    )
    parser.add_argument(
        "--model_path",
        default=None,
    )
    args = parser.parse_args()
    return args


def main():
    inputs = parse_args()
    env = inputs.env
    model_path = inputs.model_path
    if env == "kaggle":
        config_path = Path(
            "/kaggle/input/bird-repo/BirdCLEF2023/config/config.yaml"
        )
    else:
        config_path = Path("config/config.yaml")

    with config_path.open() as fh:
        config = yaml.safe_load(fh)

    if model_path is None:
        model_path = config[env]["predict"]["model_path"]
    model_path = Path(model_path)
    hparams_path = model_path / "hparams.yaml"
    state_dict_path = list(model_path.glob("*.pt"))[0]

    with hparams_path.open() as fh:
        hparams = yaml.safe_load(fh)

    model = BirdModel(**hparams)
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    predict_dir = Path(config[env]["predict"]["predict_dir"])
    filenames = [file.name for file in predict_dir.glob("*.ogg")]
    dataset = PredictDataset(base_path=str(predict_dir), filenames=filenames)

    dataloader = DataLoader(
        dataset,
        batch_size=config[env]["predict"]["batch_size"],
        shuffle=False,
        collate_fn=predict_collate_fn,
        num_workers=config[env]["predict"]["num_workers"],
    )

    submission_df = pd.DataFrame(columns=["row_id"] + LABELS)
    for batch in dataloader:
        row_ids = batch["row_ids"]
        images = batch["images"]
        preds = model(images).sigmoid().detach().numpy()
        preds_dict = {"row_id": row_ids}
        preds_dict.update({label: value for label, value in zip(LABELS, preds.T)})
        preds_df = pd.DataFrame(preds_dict)
        submission_df = pd.concat(
            [submission_df, preds_df],
            ignore_index=True,
        )

    submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
