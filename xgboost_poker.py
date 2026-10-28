
from pathlib import Path
import sys

import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

archivo_entrada = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("poker-hand-training.csv")
)

if not archivo_entrada.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {archivo_entrada}"
    )

df = pd.read_csv(archivo_entrada)

# Separar las características y la clase que queremos predecir
X = df.drop(columns=["Poker Hand"])
y = df["Poker Hand"]

# Separar los datos para entrenar y validar
train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

# Crear el modelo XGBoost
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=10,
    eval_metric="mlogloss",
    random_state=0
)

# Entrenar el modelo
print("Entrenando modelo XGBoost...")
model.fit(train_X, train_y)

# Evaluar el modelo con los datos de validación
predictions = model.predict(val_X)

print("\nResultados del modelo:")
print("Registros de entrenamiento:", len(train_X))
print("Registros de validacion:", len(val_X))
print("Accuracy:", accuracy_score(val_y, predictions))
print("MAE:", mean_absolute_error(val_y, predictions))

archivo_modelo = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path("model") / "xgboost_poker.joblib"
)

archivo_modelo.parent.mkdir(
    parents=True,
    exist_ok=True
)

dump(model, archivo_modelo)

print("\nModelo guardado correctamente.")
print("Ubicacion:", archivo_modelo)
