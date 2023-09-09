from fastapi import FastAPI
from fastapi.param_functions import Depends
from pydantic import BaseModel

from models.tft import TemporalFusionTransformer

app = FastAPI()


class CustomForm(BaseModel):
    agency: str
    sku: str
    volume: float
    date: str
    industry_volume: float
    soda_volume: float
    avg_max_temp: float
    price_regular: float
    price_actual: float
    discount: float
    avg_population_2017: float
    avg_yearly_household_income_2017: float
    easter_day: int
    good_friday: int
    new_year: int
    christmas: int
    labor_day: int
    independence_day: int
    revolution_day_memorial: int
    regional_games: int
    fifa_u_17_world_cup: int
    football_gold_cup: int
    beer_capital: int
    music_fest: int
    discount_in_percent: float
    timeseries: str


@app.post("/predictions")
async def predictions(form_data: CustomForm = Depends()):
    try:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(
            "../output/savedmodels/tft.ckpt"
        )
        predictions = best_tft.predict(form_data)
        return predictions
    except Exception as e:
        return e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
