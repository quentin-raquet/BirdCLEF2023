# BirdCLEF2023

BirdCLEF2023 is a project designed to participate in the BirdCLEF 2023 competition on Kaggle.
The goal of this competition is to classify bird songs from audio files.
https://www.kaggle.com/competitions/birdclef-2023

## Project Structure

The project is structured as follows:

- `bird/`: Contains the main source code for the project.
  - `config/`: Contains configuration files, including `config.yaml` and `labels.py`.
  - `src/`: Contains the main Python scripts for the project, including `data.py`, `metric.py`, `model.py`, and `preprocessing.py`.
- `data/`: Contains the audio data for the project, split into `test_soundscapes/` and `train_audio/`.
- `model/`: Contains the trained model files.
- `predict.py`: Script for making predictions with the trained model.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `setup.py`: Script for setting up the project.
- `train_databricks.py`: Script for training the model on Databricks.
- `train.py`: Script for training the model.

## How to Run

To train the model, run:

```sh
python train.py
```

To make predictions with the trained model, run:

```sh
python predict.py
```
