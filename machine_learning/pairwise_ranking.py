from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb


def train_test_split_and_modelling(df):
    gss = GroupShuffleSplit(test_size=.40, n_splits=1, random_state=7).split(df, groups=df['ID'])

    X_train_inds, X_test_inds = next(gss)

    train_data = df.iloc[X_train_inds]
    X_train = train_data.loc[:, ~train_data.columns.isin(['ID', 'RES21'])]  # Exclude 'ID' and 'RES21' columns
    y_train = train_data.loc[:, 'RES21']  # Only 'RES21' column

    groups = train_data.groupby('ID').size().to_frame('size')['size'].to_numpy()
    # groups = train_data.groupby('ID').size().to_numpy()

    test_data = df.iloc[X_test_inds]

    # We need to keep the id for later predictions
    X_test = test_data.loc[:]
    y_test = test_data.loc[:, 'RES21']

    model = xgb.XGBRanker(
        objective='rank:pairwise',
        random_state=42,
        learning_rate=0.1,
        colsample_bytree=0.9,
        eta=0.05,
        max_depth=6,
        n_estimators=110,
        subsample=0.75
    )

    model.fit(X_train, y_train, group=groups, verbose=True)
    return model, X_test, y_test


def predict(model, df):
    return model.predict(df)


def rank_feature(predictions):
    # Assuming predictions is the output from start_ranking(df)
    ranked_samples = {}
    for group, pred_scores in predictions.items():
        sorted_indices = pred_scores.argsort()[::1]  # Sort indices in escending order
        ranked_samples[group] = sorted_indices  # Start ranking from 1

    return ranked_samples


def start_ranking(df):
    # Train ranking model
    model, X_test, y_test = train_test_split_and_modelling(df)

    # Predict
    predictions = (X_test.groupby('ID')
                   .apply(lambda x: predict(model, x.drop(columns=['ID', 'RES21']))))

    # print(rank_feature(predictions))
    ranked_samples = rank_feature(predictions)

    # Add original RES21 values to the ranked samples
    for group, indices in ranked_samples.items():
        ranked_samples[group] = {
            'predicted_positions': indices + 1,  # Start ranking from 1
            'original_RES21': df.loc[df['ID'] == group, 'RES21'].values
        }

    return ranked_samples


def pairwise_learn_to_rank(df, target):
    result = start_ranking(df)

    for group, data in result.items():
        print("Group ID:", group)
        print("Predicted Positions:", data['predicted_positions'])
        print("Original RES21 Values:", data['original_RES21'])

    return result
