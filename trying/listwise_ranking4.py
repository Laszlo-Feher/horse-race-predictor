

import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

def compare_positions(predictions, test_y):
    correct_predictions = 0

    for pred, true in zip(predictions, test_y):
        # Az első helyezett indexének meghatározása a valós értékek között
        true_first_place_index = true.idxmin()

        # Az első helyezett indexének meghatározása a predikciók között
        pred_first_place_index = pred.argmin()

        # Összehasonlítás az első helyezett indexének alapján
        if true_first_place_index == pred_first_place_index:
            correct_predictions += 1

    return correct_predictions

def convert_to_positions(predictions):
    positions = []
    for pred in predictions:
        # Ha csak egy dimenziós a predikciók tömbje
        if pred.ndim == 1:
            first_place_index = pred.argmin()
            second_place_index = None
        else:
            # Pontszámok rangsorolása
            ranked_positions = pred.argsort(axis=0)
            # Az első helyezett pozíciója (a legkisebb pontszámhoz tartozó index)
            first_place_index = ranked_positions[0]
            # A második helyezett pozíciója (a második legkisebb pontszámhoz tartozó index)
            second_place_index = ranked_positions[1]

        # Az eredmények listába gyűjtése
        positions.append((first_place_index, second_place_index))

    return positions


def listwise_learn_to_rank2(df):
    # Adattípusok és DataFrame előkészítése
    df['ID'] = df['ID'].astype(str)
    df['RES21'] = df['RES21'].astype(float)

    # Kiválasztjuk a bemeneti és kimeneti változókat
    X = df.drop(columns=['RES21'])
    y = df['RES21']

    # DataFrame szétválasztása tanító és tesztelő adatokra versenyenként
    train_X, test_X, train_y, test_y = [], [], [], []
    for race_id, group in df.groupby('ID'):
        group_X = group.drop(columns=['ID'])
        group_y = group['RES21']
        X_train, X_test, y_train, y_test = train_test_split(group_X, group_y, test_size=0.2, random_state=42)
        train_X.append(X_train)
        test_X.append(X_test)
        train_y.append(y_train)
        test_y.append(y_test)

    # Modell létrehozása
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(X.columns),)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Modell betanítása minden versenyhez
    for X_train, y_train in zip(train_X, train_y):
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predikció futtatása minden teszt adathalmazra
    predictions = []
    for X_test in test_X:
        predictions.append(model.predict(X_test))

    mse_scores = []
    for pred, true in zip(predictions, test_y):
        mse = mean_squared_error(true, pred)
        mse_scores.append(mse)

    print(predictions, test_y)
    print(mse_scores)
    average_mse = sum(mse_scores) / len(mse_scores)
    print("Átlagos négyzetes hiba:", average_mse)

    # converted_predictions = convert_to_positions(predictions)
    # print("Converted predictions:", converted_predictions)

    # correct_predictions_count = compare_positions(predictions, test_y)
    # print("Helyes predikciók az első helyre vonatkozóan:", correct_predictions_count)

    return predictions, test_y

def run_listwise(df):
    listwise_learn_to_rank2(df)
