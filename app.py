
import hmac
import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURACION DEL MODELO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "xgboost_poker.joblib"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"No se encontro el modelo entrenado: {MODEL_PATH}"
    )

model = load(MODEL_PATH)


# ============================================================
# DATOS DEL MODELO DE POKER
# ============================================================

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
    "Rank of Card 5",
]

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
    9: "Escalera real",
}


# ============================================================
# CONFIGURACION DE GITHUB ACTIONS
# ============================================================

GITHUB_DISPATCH_URL = (
    "https://api.github.com/repos/"
    "IanJS7/machine-learning-poker/dispatches"
)

ALLOWED_DATASETS = {
    "https://res.cloudinary.com/lq5tulzu/raw/upload/v1793156193/poker-hand-training-25.csv",
    "https://res.cloudinary.com/lq5tulzu/raw/upload/v1793156194/poker-hand-training-50.csv",
    "https://res.cloudinary.com/lq5tulzu/raw/upload/v1793156194/poker-hand-training-100.csv",
}


# ============================================================
# CREACION DE LA API
# ============================================================

app = FastAPI(
    title="Poker ModelOps API",
    description=(
        "API para predecir manos de poker y solicitar "
        "el reentrenamiento de XGBoost mediante GitHub Actions."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELOS DE ENTRADA Y SALIDA
# ============================================================

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


class OutputData(BaseModel):
    prediction: int
    hand_name: str


class RetrainRequest(BaseModel):
    dataseturl: str
    sha: str = Field(min_length=1, max_length=128)


# ============================================================
# RUTA PRINCIPAL
# ============================================================

@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "Poker ModelOps API",
        "docs": "/docs",
        "endpoints": ["/score", "/retrain"],
    }


# ============================================================
# PREDICCION DE MANOS DE POKER
# ============================================================

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
        data.rank_card_5,
    ]]

    model_input = pd.DataFrame(
        values,
        columns=FEATURES,
    )

    prediction = int(model.predict(model_input)[0])

    return {
        "prediction": prediction,
        "hand_name": HAND_NAMES.get(
            prediction,
            "Clase desconocida",
        ),
    }


# ============================================================
# SOLICITUD SEGURA DE REENTRENAMIENTO
# ============================================================

@app.post("/retrain")
def retrain(
    data: RetrainRequest,
    x_retrain_key: str | None = Header(
        default=None,
        alias="X-Retrain-Key",
    ),
):
    # Estas variables se configuran en el entorno de Render.
    github_token = os.getenv("GITHUB_TOKEN")
    expected_key = os.getenv("RETRAIN_API_KEY")

    if not github_token or not expected_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "El servicio de reentrenamiento no esta "
                "configurado en Render."
            ),
        )

    # Comprobar la clave de acceso enviada desde la interfaz.
    if (
        not x_retrain_key
        or not hmac.compare_digest(
            x_retrain_key,
            expected_key,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Clave de acceso incorrecta.",
        )

    # Solo se aceptan las tres URL autorizadas de Cloudinary.
    if data.dataseturl not in ALLOWED_DATASETS:
        raise HTTPException(
            status_code=400,
            detail="El dataset seleccionado no esta permitido.",
        )

    # Validar la etiqueta que se utilizara para la imagen Docker.
    tag = data.sha.strip()

    if not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}",
        tag,
    ):
        raise HTTPException(
            status_code=400,
            detail="La etiqueta de Docker no es valida.",
        )

    # Payload esperado por el workflow de GitHub Actions.
    payload = {
        "event_type": "ml_ci_cd",
        "client_payload": {
            "dataseturl": data.dataseturl,
            "sha": tag,
        },
    }

    github_request = Request(
        GITHUB_DISPATCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "Poker-ModelOps",
        },
        method="POST",
    )

    try:
        with urlopen(github_request, timeout=20) as response:
            if response.status != 204:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "GitHub no acepto el evento de reentrenamiento."
                    ),
                )

        return {
            "status": "dispatched",
            "message": (
                "Solicitud enviada a GitHub Actions. "
                "Consulta la seccion Actions para ver el progreso."
            ),
            "tag": tag,
        }

    except HTTPError as error:
        # No devolver al cliente informacion sensible del token.
        raise HTTPException(
            status_code=502,
            detail=(
                f"GitHub rechazo la solicitud con HTTP {error.code}. "
                "Verifica GITHUB_TOKEN y los permisos del repositorio."
            ),
        ) from error

    except URLError as error:
        raise HTTPException(
            status_code=502,
            detail="No fue posible contactar con GitHub.",
        ) from error
