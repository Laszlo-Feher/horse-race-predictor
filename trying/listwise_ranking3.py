# import tensorflow as tf
# import tensorflow_ranking as tfr
# import pandas as pd
#
# def train_and_predict(df, num_features, hidden_units, num_epochs, window_size):
#     # Define the model
#     class ListwiseLTRModel(tf.keras.Model):
#         def __init__(self, num_features, hidden_units):
#             super(ListwiseLTRModel, self).__init__()
#             self.racer_embedding = tf.keras.layers.Embedding(input_dim=df["ID"].max() + 1, output_dim=hidden_units)
#             self.feature_layers = [tf.keras.layers.Dense(hidden_units, activation="relu") for _ in range(num_features)]
#             self.score_model = tf.keras.layers.Dense(1)
#
#         def call(self, inputs):
#             racer_embedding = self.racer_embedding(inputs["ID"])
#             features = tf.stack([self.feature_layers[i](inputs[f"E{i+1}"]) for i in range(num_features)], axis=1)
#             concatenated_embeddings = tf.concat([racer_embedding, features], axis=1)
#             scores = self.score_model(concatenated_embeddings)
#             return scores
#
#         def compute_loss(self, features, labels, training=False):
#             scores = self(features)
#             return tfr.keras.losses.ListMLELoss()(labels, scores)
#
#     # Compile the model
#     model = ListwiseLTRModel(num_features, hidden_units)
#     model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1), loss=model.compute_loss)
#
#     # Create a dataset from your dataframe
#     dataset = tf.data.Dataset.from_tensor_slices((df.to_dict(orient='list'), df["RES21"].values))
#
#     # Group the dataset by race ID
#     dataset = dataset.window(window_size, drop_remainder=True)
#     dataset = dataset.flat_map(lambda x: x.batch(window_size))
#     dataset = dataset.map(lambda x, y: ({"ID": x["ID"], "E1": x["E1"], "E2": x["E2"]}, y))
#
#     # Train the model
#     model.fit(dataset, epochs=num_epochs, verbose=False)
#
#     # Define a method for making predictions
#     def predict(inputs):
#         racer_embedding = model.racer_embedding(tf.constant([inputs["ID"]]))
#         features = tf.stack([model.feature_layers[i](tf.constant(inputs[f"E{i+1}"])) for i in range(num_features)], axis=1)
#         concatenated_embeddings = tf.concat([racer_embedding, features], axis=1)
#         scores = model.score_model(concatenated_embeddings)
#         return scores.numpy()
#
#     return predict
#
#
# def run_listwise(df, target, formatted_time):
#     # Example usage
#     # df = pd.read_csv("racer_data.csv")  # replace with your actual data file
#     predict = train_and_predict(df, num_features=50, hidden_units=128, num_epochs=30, window_size=10)
#
#     print(predict)
#     return None
# #
# from sklearn.metrics import mean_squared_error
# import math
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
#
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
#
# def preprocess_data(df):
#     # Normalize numerical features
#     numeric_cols = df.select_dtypes(include=np.number).columns
#     df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
#
#     # Convert categorical features to one-hot encoding
#     # df = pd.get_dummies(df)
#
#     return df
#
# def split_data(df):
#     # Group by ID and split into train-test
#     ids = df['ID'].unique()
#     train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)
#     train_df = df[df['ID'].isin(train_ids)]
#     test_df = df[df['ID'].isin(test_ids)]
#
#     X_train, y_train = train_df.drop(columns=['RES21']), train_df['RES21']
#     X_test, y_test = test_df.drop(columns=['RES21']), test_df['RES21']
#
#     return X_train, X_test, y_train, y_test, test_df
#
# def build_model(input_shape):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(1)  # Output layer
#     ])
#
#     model.compile(optimizer='adam', loss='mean_squared_error')
#
#     return model
#
# from sklearn.metrics import mean_squared_error, mean_absolute_error
#
# def evaluate_predictions(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#
#     print(f'Mean Squared Error (MSE): {mse}')
#     print(f'Mean Absolute Error (MAE): {mae}')
#
#     return mse, mae
#
# # Example usage:
#
# def predict_race(model, X_race):
#     X_race = preprocess_data(X_race)  # Preprocess data if needed
#     X_race = X_race.drop(columns=['ID', 'RES21'])  # Drop unnecessary columns
#     predictions = model.predict(X_race)  # Make predictions
#     return predictions
#
# def predict(model, df):
#     return model.predict(df)
#
# def run_listwise(df):
#     # Preprocess data
#     # df = preprocess_data(df)
#
#     # Split data
#     X_train, X_test, y_train, y_test, test_df = split_data(df)
#
#     # Build model
#     print(X_train)
#     model = build_model(input_shape=(X_train.shape[1],))
#
#     # Train model
#     model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
#
#     # Evaluate model
#     print(X_test)
#     test_loss = model.evaluate(X_test, y_test, verbose=2)
#     print(f'Test Loss: {test_loss}')
#
#     # Make predictions
#     # predictions = model.predict(X_test)
#     # print(len(y_test), len(predictions))
#     print(test_df)
#     predictions = (test_df.groupby('ID')
#                    .apply(lambda x: predict(model, x.drop(columns=['ID', 'RES21']))))
#
#     print(predictions)
#     # mse, mae = evaluate_predictions(y_test, predictions)
#
#     # print(mse, mae)
#
#     return predictions
#
# # Assuming df is your DataFrame
#

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

    correct_predictions_count = compare_positions(predictions, test_y)
    print("Helyes predikciók az első helyre vonatkozóan:", correct_predictions_count)

    return predictions, test_y

def run_listwise(df):
    listwise_learn_to_rank2(df)
