import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import kendalltau, spearmanr
import math

from io_utils.file_writer import create_text_file, write_to_file

def create_model():
    model = tf.keras.Sequential()
    num_features = 38
    model.add(layers.Dense(64, activation='relu', input_shape=(num_features,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_test_split_and_modelling(df):
    gss = GroupShuffleSplit(test_size=.40, n_splits=1, random_state=7).split(df, groups=df['ID'])

    X_train_inds, X_test_inds = next(gss)

    train_data = df.iloc[X_train_inds]
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    X_train = train_data.loc[:, ~train_data.columns.isin(['ID', 'RES21'])]  # Exclude 'ID' and 'RES21' columns
    y_train = train_data.loc[:, 'RES21']  # Only 'RES21' column
    print(y_train)
    groups = train_data.groupby('ID').size().to_frame('size')['size'].to_numpy()

    test_data = df.iloc[X_test_inds]
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    # We need to keep the id for later predictions
    X_test = test_data.loc[:]
    y_test = test_data.loc[:, 'RES21']

    model = create_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model, X_test, y_test

def predict(model, df):
    return model.predict(df)

# def predict(model, df):
#     predictions = []
#     for i in range(df.shape[0]):
#         row = df.iloc[i]
#         prediction = model.predict(row.values.reshape(1, -1))
#         predictions.append(prediction[0])
#     return np.array(predictions)

def rank_feature(predictions):
    # Assuming predictions is the output from start_ranking(df)
    ranked_samples = {}
    for group, pred_scores in predictions.items():
        sorted_indices = pred_scores.argsort()[::1]  # Sort indices in escending order
        ranked_samples[group] = sorted_indices  # Start ranking from 1

    return ranked_samples

def start_ranking_list(df):
    # Train ranking model
    model, X_test, y_test = train_test_split_and_modelling(df)

    # Predict
    predictions = (X_test.groupby('ID')
                   .apply(lambda x: predict(model, x.drop(columns=['ID', 'RES21']))))

    # print(rank_feature(predictions))
    # print(predictions)
    # averages = np.mean(predictions[2], axis=1)
    # print(averages)
    # new_predictions = []
    # for arr in predictions: # Minden tömbön elvégzi a predikciót, kivéve az utolsó oszlopot
    #     new_predictions.append(np.mean(arr))
    # for key, value in predictions.items():
    #     averages = [sum(sublist) / len(sublist) for sublist in value]
    #     predictions[key] = averages
    # print(new_predictions)
    all_predictions = [pred for sublist in predictions for pred in sublist]

    print(predictions)
    ranked_samples = rank_feature(predictions)
    print(ranked_samples)

    # Add original RES21 values to the ranked samples
    for group, indices in ranked_samples.items():
        ranked_samples[group] = {
            'predicted_positions': indices + 1,  # Start ranking from 1
            'original_RES21': df.loc[df['ID'] == group, 'RES21'].values
        }
    print(ranked_samples)

    return ranked_samples

# Rest of the functions remain the same
