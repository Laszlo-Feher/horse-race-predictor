# import pandas as pd
# import numpy as np
# from ranklib import RankLib
#
# # Assuming df is your DataFrame
# # Convert the DataFrame to a listwise format
# listwise_data = []
# for _, group in df.groupby('ID'):
#     listwise_data.append(list(group.drop('ID', axis=1).values))
#
# # Save the listwise data to a file
# np.savetxt("listwise_data.txt", listwise_data, fmt="%f", delimiter="\t")
#
# # Initialize RankLib
# ranklib = RankLib()
#
# # Set the parameters for ListNet
# ranklib.setParameter("list", "ListNet")
# ranklib.setParameter("C", "1.0")
#
# # Train the listwise model
# ranklib.train("listwise_data.txt", "listnet_model.txt")
#
# # Predict the results for a new race
# new_race = [
#     [1.0, 2018.0, 9.0, 0.0, 0.0, 2.0, 46.6, 88.0, 73.0, 41.0],
#     [1.0, 2018.0, 4.0, 0.0, 0.0, 1.0, 46.6, 88.0, 73.0, 9.9],
#     [1.0, 2018.0, 1.0, 0.0, 0.0, 0.0, 46.0, 72.8, 60.0, 22.3],
#     [1.0, 2018.0, 1.0, 0.0, 0.0, 1.0, 74.2, 100.6, 23.8, 8.9]
# ]
#
# predictions = ranklib.index("listwise_data.txt", new_race, "listnet_model.txt")
# print(predictions)

import tensorflow as tf
import tensorflow_ranking as tfr
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from keras import layers, models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


# Define RankNet model architecture
def create_ranknet_model(input_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')  # Output layer for ranking scores
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')  # Use appropriate loss for LTR
    return model

# Define the model
class ListwiseLTRModel(tf.keras.Model):
    def __init__(self, num_features, hidden_units):
        super(ListwiseLTRModel, self).__init__()
        self.racer_embedding = tf.keras.layers.Embedding(input_dim=df["ID"].max() + 1, output_dim=hidden_units)
        self.feature_layers = tf.keras.layers.Dense(hidden_units, activation="relu")
        self.score_model = tf.keras.layers.Dense(1)

    def call(self, inputs):
        racer_embedding = self.racer_embedding(inputs["racer_id"])
        features = self.feature_layers(inputs["feature1"], inputs["feature2"],...)  # add your features here
        concatenated_embeddings = tf.concat([racer_embedding, features], axis=1)
        scores = self.score_model(concatenated_embeddings)
        return scores

    def compute_loss(self, features, labels, training=False):
        scores = self(features)
        return tfr.keras.losses.ListMLELoss()(labels, scores)

def train_test_split_and_modelling(df):
    # gss = GroupShuffleSplit(test_size=.40, n_splits=1, random_state=7).split(df, groups=df['ID'])
    #
    # X_train_inds, X_test_inds = next(gss)
    #
    # train_data = df.iloc[X_train_inds]
    # train_data = train_data.sample(frac=1).reset_index(drop=True)
    #
    # X_train = train_data.loc[:, ~train_data.columns.isin(['ID', 'RES21'])]  # Exclude 'ID' and 'RES21' columns
    # y_train = train_data.loc[:, 'RES21']  # Only 'RES21' column
    #
    # groups = train_data.groupby('ID').size().to_frame('size')['size'].to_numpy()
    # # groups = train_data.groupby('ID').size().to_numpy()
    #
    # test_data = df.iloc[X_test_inds]
    # test_data = test_data.sample(frac=1).reset_index(drop=True)
    #
    # # We need to keep the id for later predictions
    # X_test = test_data.loc[:]
    # y_test = test_data.loc[:, 'RES21']
    #
    # # Train RankNet model
    # ranknet_model = create_ranknet_model(input_dim=X_train.shape[1])
    # ranknet_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    #
    # # Evaluate the model
    # evaluation = ranknet_model.evaluate(X_test, y_test)
    # X_train, X_test, y_train, y_test = train_test_split(df.drop('RES21', axis=1), df['RES21'], test_size=0.2, random_state=42)
    #
    # # Define your model
    # model = Sequential()
    # model.add(Dense(128, activation='relu', input_shape=(40,)))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(1, activation='linear'))
    #
    # # Compile your model
    # model.compile(optimizer='adam', loss='mse')
    #
    # # Train your model
    # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    # evaluation = model.fit(X_test, y_test)
    # print(evaluation)
    # # Predict ranking for new races
    # # predictions = ranknet_model.predict(new_race_data)


    # Compile the model
    model = ListwiseLTRModel(num_features=10, hidden_units=128)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    # Create a dataset from your dataframe
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), df["RES21"]))

    # Group the dataset by race ID
    dataset = dataset.group_by_window(lambda x: x["ID"], window_size=10, drop_remainder=True)

    # Create a listwise dataset
    listwise_dataset = dataset.map(lambda x: {"features": x, "labels": x["RES21"]})

    # Train the model
    model.fit(listwise_dataset, epochs=30, verbose=False)


def run_listwise(df, target, formatted_time):
    train_test_split_and_modelling(df)
    return None
