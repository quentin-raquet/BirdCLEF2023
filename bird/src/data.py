import pandas as pd
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.preprocessing import Preprocessing
from config.labels import LABELS

from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule


class BirdDataset(Dataset):
    def __init__(
        self,
        base_path,
        filenames,
        labels,
        secondary_labels,
        label_list=LABELS,
        nb_samples=5,
        split_method="random",
        resample_freq=32_000,
        duration=5,
        n_fft=2_048,
        n_mels=128,
        hop_length=512,
        top_db=80,
    ):
        self.base_path = Path(base_path)
        self.filenames = filenames
        one_hot_labels = []
        for label, secondary_label in zip(labels, secondary_labels):
            one_hot_label = torch.zeros(len(label_list))
            one_hot_label[label_list.index(label)] = 1.0
            for sl in eval(secondary_label):
                one_hot_label[label_list.index(sl)] = 1.0
            one_hot_labels.append(one_hot_label)
        self.labels = torch.stack(one_hot_labels)
        self.nb_samples = nb_samples
        self.transform = Preprocessing(
            resample_freq=resample_freq,
            duration=duration,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            top_db=top_db,
            nb_samples=nb_samples,
            split_method=split_method,
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        waveform, sample_rate = torchaudio.load(self.base_path / filename)
        image = self.transform(waveform, sample_rate)
        label = self.labels[idx, :].unsqueeze(0).repeat(image.size(0), 1)
        return image, label


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return {
        "images": torch.cat(data).unsqueeze(1).float(),
        "labels": torch.cat(target).float(),
    }


class BirdDataModule(LightningDataModule):
    def __init__(
        self,
        metadata_path: str,
        train_dir: str,
        valid_size: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 32,
        nb_samples=5,
        resample_freq=32_000,
        duration=5,
        n_fft=2_048,
        n_mels=128,
        hop_length=512,
        top_db=80,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nb_samples = nb_samples
        self.resample_freq = resample_freq
        self.duration = duration
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.top_db = top_db

        df = pd.read_csv(metadata_path)

        label_cnt = df.primary_label.value_counts()

        # Get labels with only one occurence
        only_one_sample = label_cnt[label_cnt == 1].index.to_list()

        # Get list of labels and filenames
        labels = df.loc[
            ~df.primary_label.isin(only_one_sample), "primary_label"
        ].to_list()
        secondary_labels = df.loc[
            ~df.primary_label.isin(only_one_sample), "secondary_labels"
        ].to_list()
        filenames = df.loc[
            ~df.primary_label.isin(only_one_sample), "filename"
        ].to_list()

        # Perform train / valid split
        (
            self.train_filenames,
            self.valid_filenames,
            self.train_labels,
            self.valid_labels,
            self.train_secondary_labels,
            self.valid_secondary_labels
        ) = train_test_split(
            filenames, labels, secondary_labels, test_size=valid_size, random_state=1992, stratify=labels
        )

        only_one_labels = df.loc[
            df.primary_label.isin(only_one_sample), "primary_label"
        ].to_list()
        only_one_secondary_labels = df.loc[
            df.primary_label.isin(only_one_sample), "secondary_labels"
        ].to_list()
        only_one_filenames = df.loc[
            df.primary_label.isin(only_one_sample), "filename"
        ].to_list()

        self.train_filenames += only_one_filenames
        self.train_labels += only_one_labels
        self.train_secondary_labels += only_one_secondary_labels

    def setup(self, stage: str = None):
        self.train_dataset = BirdDataset(
            base_path=self.train_dir,
            filenames=self.train_filenames,
            labels=self.train_labels,
            secondary_labels=self.train_secondary_labels,
            nb_samples=self.nb_samples,
            split_method="random",
            resample_freq=self.resample_freq,
            duration=self.duration,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            top_db=self.top_db,
        )
        self.valid_dataset = BirdDataset(
            base_path=self.train_dir,
            filenames=self.valid_filenames,
            labels=self.valid_labels,
            secondary_labels=self.valid_secondary_labels,
            nb_samples=self.nb_samples,
            split_method="ordered",
            resample_freq=self.resample_freq,
            duration=self.duration,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            top_db=self.top_db,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class PredictDataset(Dataset):
    def __init__(
        self,
        base_path,
        filenames,
        nb_samples=None,
        split_method="ordered",
        resample_freq=32_000,
        duration=5,
        n_fft=2_048,
        n_mels=128,
        hop_length=512,
        top_db=80,
    ):
        self.base_path = Path(base_path)
        self.filenames = filenames
        self.duration = duration
        self.transform = Preprocessing(
            resample_freq=resample_freq,
            duration=duration,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            top_db=top_db,
            nb_samples=nb_samples,
            split_method=split_method,
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        waveform, sample_rate = torchaudio.load(self.base_path / filename)
        images = self.transform(waveform, sample_rate)
        filename = filename.split(".")[0]
        row_ids = [
            f"{filename}_{i*self.duration}" for i in range(1, images.size(0) + 1)
        ]
        return row_ids, images


def predict_collate_fn(batch):
    row_ids = [item for sublist in batch for item in sublist[0]]
    images = [item[1] for item in batch]
    return {
        "row_ids": row_ids,
        "images": torch.cat(images).unsqueeze(1).float(),
    }
