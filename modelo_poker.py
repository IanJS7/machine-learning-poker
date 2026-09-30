import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("poker-hand-training.csv")

features = [
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

X = df[features]
y = df["Poker Hand"]

train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

model = DecisionTreeClassifier(random_state=0)
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)

print("Registros de entrenamiento:", len(train_X))
print("Registros de validación:", len(val_X))
print("Accuracy:", accuracy_score(val_y, val_predictions))
