import pickle
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_FILE = 'model.bin'

with open(MODEL_FILE, 'rb') as f_in:
    dv, model, age_median = pickle.load(f_in)

app = FastAPI(title='mammographic-mass-classification')


class Mass(BaseModel):
    # any field can be omitted if unknown; missing values are handled as in training
    age: Optional[float] = Field(None, ge=0, le=120)
    shape: Optional[int] = Field(None, ge=1, le=4)      # round=1, oval=2, lobular=3, irregular=4
    margin: Optional[int] = Field(None, ge=1, le=5)     # circumscribed=1 ... spiculated=5
    density: Optional[int] = Field(None, ge=1, le=4)    # high=1, iso=2, low=3, fat-containing=4


def prepare(mass: Mass) -> dict:
    return {
        'age': age_median if mass.age is None else mass.age,
        'shape': 'unknown' if mass.shape is None else str(mass.shape),
        'margin': 'unknown' if mass.margin is None else str(mass.margin),
        'density': 'unknown' if mass.density is None else str(mass.density),
    }


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(mass: Mass):
    X = dv.transform([prepare(mass)])
    probability = float(model.predict_proba(X)[0, 1])
    return {'malignancy_probability': round(probability, 4)}