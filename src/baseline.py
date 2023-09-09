import warnings

import torch
from pytorch_forecasting import Baseline
from pytorch_forecasting.metrics import MAE

from dataloader import get_dataloaders, load_data


def train_baseline(val_dataloader) -> torch.Tensor:
    baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    return MAE()(baseline_predictions.output, baseline_predictions.y)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data = load_data("../data/processed/stallion.parquet")
    train_dataloader, val_dataloader, train_dataset, val_dataset = get_dataloaders(data)
    print(train_baseline(val_dataloader).item())
