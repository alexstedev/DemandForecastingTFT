import warnings

warnings.filterwarnings("ignore")

from pytorch_forecasting.metrics import MAE

from dataloader import get_dataloaders, load_data
from models.tft import TemporalFusionTransformer

data = load_data("../data/processed/stallion.parquet")
train_dataloader, val_dataloader, training, validation = get_dataloaders(data)

# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_tft = TemporalFusionTransformer.load_from_checkpoint(
    "../output/savedmodels/tft.ckpt"
)

# calculate mean absolute error on validation set
predictions = best_tft.predict(
    val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
)
prediction_mae = MAE()(predictions.output, predictions.y)

print(prediction_mae)
