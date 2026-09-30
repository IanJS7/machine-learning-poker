import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("poker-hand-training.csv")

X = df.drop(columns=["Poker Hand"])
y = df["Poker Hand"]

train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

model = DecisionTreeClassifier(random_state=0)
model.fit(train_X, train_y)

train_predictions = model.predict(train_X)
val_predictions = model.predict(val_X)

print("Registros de entrenamiento:", len(train_X))
print("Registros de validacion:", len(val_X))
print("Accuracy en entrenamiento:", accuracy_score(train_y, train_predictions))
print("Accuracy en validacion:", accuracy_score(val_y, val_predictions))
