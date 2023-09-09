import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from dataloader import load_data
from models.tft import TemporalFusionTransformer

data = load_data("../data/processed/stallion.parquet")

best_tft = TemporalFusionTransformer.load_from_checkpoint(
    "../output/savedmodels/tft.ckpt"
)

max_encoder_length = 24
max_prediction_length = 6

# select last 24 months from data (max_encoder_length is 24)
encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

# select last known data point and create decoder data from it by repeating it and incrementing the month
# in a real world dataset, we should not just forward fill the covariates but specify them to account
# for changes in special days and prices (which you absolutely should do but we are too lazy here)
last_data = data[lambda x: x.time_idx == x.time_idx.max()]
decoder_data = pd.concat(
    [
        last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i))
        for i in range(1, max_prediction_length + 1)
    ],
    ignore_index=True,
)

# add time index consistent with "data"
decoder_data["time_idx"] = (
    decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
)
decoder_data["time_idx"] += (
    encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()
)

# adjust additional time feature(s)
decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype(
    "category"
)  # categories have be strings

# combine encoder and decoder data
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

new_raw_predictions = best_tft.predict(new_prediction_data, return_x=True)

new_raw_predictions.to_csv("../output/predictions/tft_predictions.csv", index=False)
