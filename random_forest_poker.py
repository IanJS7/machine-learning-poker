import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("poker-hand-training.csv")

X = df.drop(columns=["Poker Hand"])
y = df["Poker Hand"]

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(train_X, train_y)

predictions = model.predict(val_X)

print("Registros de entrenamiento:", len(train_X))
print("Registros de validacion:", len(val_X))
print("Accuracy:", accuracy_score(val_y, predictions))
print("MAE:", mean_absolute_error(val_y, predictions))
