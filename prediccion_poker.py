import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv("poker-hand-training.csv")
test_data = pd.read_csv("poker-hand-testing.csv")

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

X_train = train_data[features]
y_train = train_data["Poker Hand"]

X_test = test_data[features]
y_test = test_data["Poker Hand"]

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Valores X utilizados para entrenar:")
print(X_train)

print("\nPrimeros 5 valores X del dataset a predecir:")
print(X_test.head())

mae = mean_absolute_error(y_test, predictions)

print("\nMAE:", mae)

comparison = pd.DataFrame({
    "Valor calculado": predictions[:5],
    "Valor real": y_test.iloc[:5].values
})

print("\nTabla comparativa:")
print(comparison.to_string(index=False))
