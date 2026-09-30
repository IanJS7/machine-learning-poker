# Machine Learning Poker

## Autor

Ian

## Descripcion

Implementacion de un modelo de aprendizaje automatico para clasificar manos de poker mediante un arbol de decision.

## Dataset

Se utilizaron los archivos:

- poker-hand-training.csv
- poker-hand-testing.csv

## Modelo

Se utilizo DecisionTreeClassifier de scikit-learn.

## Validacion

Se utilizo train_test_split con random_state=0 y test_size=0.2.

Resultados obtenidos:

- Registros de entrenamiento: 20008
- Registros de validacion: 5002
- Accuracy en entrenamiento: 1.0
- Accuracy en validacion: 0.4868052778888445

Tambien se realizo la prueba con la division predeterminada de train_test_split:

- Registros de entrenamiento: 18757
- Registros de validacion: 6253
- Accuracy en validacion: 0.4866464097233328

## Ejecucion

python modelo_poker.py
python validacion_poker.py
