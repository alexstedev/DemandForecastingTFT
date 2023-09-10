<h1 align = "center">
  Demand forecasting with Temporal Fusion Transformers <br>
</h1>

---

This repository contains a custom implementation of [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf) using Lightning and pytorch_forecasting for demand forecasting on the [Stallion dataset](https://www.kaggle.com/datasets/utathya/future-volume-prediction).

## Technical details

![](./static/images/tft_arch.png)

Enhancements compared to the original implementation in [the Google Research repo](https://github.com/google-research/google-research/tree/master/tft):

* capabilities added through pytorch_forecasting base model e.g. monotone constraints
* static variables can be continuous
* multiple categorical variables can be summarized with an EmbeddingBag
* variable encoder and decoder length by sample
* categorical embeddings are not transformed by variable selection network (because it is a redundant operation)
* variable dimension in variable selection network are scaled up via linear interpolation to reduce
  number of parameters
* non-linear variable processing in variable selection network can be shared among decoder and encoder
  (not shared by default)

## Run locally

The dependency management system is poetry. Install poetry and run:

```bash
poetry install
```

to set up the environment.

### Training 

You can get the baseline performance in the dataset by training pytorch_forecasting's Baseline model:

```bash
cd src
poetry run python baseline.py
```

To train the full model and get the performance:

```bash
poetry run python train.py
poetry run python evaluate.py
```

### Prediction & Inference

To run prediction on test data in the model, run:

```bash
poetry run python predict.py
```

To inference the model through an API, the `api.py` file sets up a simple Flask API with a '/predictions' GET endpoint.

```bash
poetry run python api.py
```

then you can access `localhost:8501/docs` to see a nice UI set up by FastAPI that allows easy inference, or use `curl` or any other tools to call the API directly.


## References

- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)
- [Stallion dataset](https://www.kaggle.com/datasets/utathya/future-volume-prediction)

