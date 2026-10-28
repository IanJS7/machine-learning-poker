
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel, Field

# Ubicacion del modelo entrenado
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "xgboost_poker.joblib"

# Cargar el modelo guardado
model = load(MODEL_PATH)

# Nombres de las caracteristicas en el orden del dataset
FEATURES = [
    "Suit of Card 1",
    "Rank of Card 1",
    "Suit of Card 2",
    "Rank of Card 2",
    "Suit of Card 3",
    "Rank of Card 3",
    "Suit of Card 4",
    "Rank of Card 4",
    "Suit of Card 5",
    "Rank of Card 5"
]

# Interpretacion de las clases del dataset
HAND_NAMES = {
    0: "Sin combinacion (carta alta)",
    1: "Un par",
    2: "Dos pares",
    3: "Tres iguales",
    4: "Escalera",
    5: "Color",
    6: "Full house",
    7: "Poker (cuatro iguales)",
    8: "Escalera de color",
    9: "Escalera real"
}

# Crear la API
app = FastAPI(
    title="Prediccion de manos de poker",
    description="API que clasifica una mano de poker utilizando XGBoost.",
    version="1.0.0"
)

# Permitir solicitudes desde diferentes origenes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Datos que recibira la API
class InputData(BaseModel):
    suit_card_1: int = Field(ge=1, le=4)
    rank_card_1: int = Field(ge=1, le=13)

    suit_card_2: int = Field(ge=1, le=4)
    rank_card_2: int = Field(ge=1, le=13)

    suit_card_3: int = Field(ge=1, le=4)
    rank_card_3: int = Field(ge=1, le=13)

    suit_card_4: int = Field(ge=1, le=4)
    rank_card_4: int = Field(ge=1, le=13)

    suit_card_5: int = Field(ge=1, le=4)
    rank_card_5: int = Field(ge=1, le=13)


# Formato de la respuesta
class OutputData(BaseModel):
    prediction: int
    hand_name: str


# Endpoint para predecir una mano
@app.post("/score", response_model=OutputData)
def score(data: InputData):
    values = [[
        data.suit_card_1,
        data.rank_card_1,
        data.suit_card_2,
        data.rank_card_2,
        data.suit_card_3,
        data.rank_card_3,
        data.suit_card_4,
        data.rank_card_4,
        data.suit_card_5,
        data.rank_card_5
    ]]

    model_input = pd.DataFrame(values, columns=FEATURES)

    prediction = int(model.predict(model_input)[0])

    return {
        "prediction": prediction,
        "hand_name": HAND_NAMES.get(prediction, "Clase desconocida")
    }
