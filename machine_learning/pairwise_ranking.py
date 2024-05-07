from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import kendalltau, spearmanr
import xgboost as xgb
import math

from io_utils.file_writer import create_text_file, write_to_file


def train_test_split_and_modelling(df, algorythm):
    gss = GroupShuffleSplit(test_size=.40, n_splits=1, random_state=7).split(df, groups=df['ID'])

    X_train_inds, X_test_inds = next(gss)

    train_data = df.iloc[X_train_inds]
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    X_train = train_data.loc[:, ~train_data.columns.isin(['ID', 'RES21'])]  # Exclude 'ID' and 'RES21' columns
    y_train = train_data.loc[:, 'RES21']  # Only 'RES21' column

    groups = train_data.groupby('ID').size().to_frame('size')['size'].to_numpy()
    # groups = train_data.groupby('ID').size().to_numpy()

    test_data = df.iloc[X_test_inds]
    test_data = test_data.sample(frac=1).reset_index(drop=True)


    # We need to keep the id for later predictions
    X_test = test_data.loc[:]
    y_test = test_data.loc[:, 'RES21']

    model = xgb.XGBRanker(
        objective=f'rank:{algorythm}',
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


def start_ranking(df, algorythm):
    # Train ranking model
    model, X_test, y_test = train_test_split_and_modelling(df, algorythm)

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


def evaluate_kendall_tau(ranked_samples):
    kendall_tau_scores = {}
    for group, data in ranked_samples.items():
        predicted_positions = data['predicted_positions']
        original_positions = data['original_RES21']

        tau, _ = kendalltau(predicted_positions, original_positions)

        kendall_tau_scores[group] = tau

    return kendall_tau_scores


def evaluate_spearman_rho(ranked_samples):
    spearman_rho_scores = {}
    for group, data in ranked_samples.items():
        predicted_positions = data['predicted_positions']
        original_positions = data['original_RES21']

        rho, _ = spearmanr(predicted_positions, original_positions)

        spearman_rho_scores[group] = rho

    return spearman_rho_scores


def calculate_average_score(scores):
    filtered_scores = {}

    for key, value in scores.items():
        if not math.isnan(value):
            filtered_scores[key] = value

    if not filtered_scores:
        return float('nan')

    total_tau = sum(filtered_scores.values())
    average_tau = total_tau / len(filtered_scores)

    return average_tau


def count_first_position_predictions(result):
    correct_first_positions = 0
    wrong_first_positions = 0

    for group, data in result.items():
        if data['predicted_positions'][0] == 1:
            correct_first_positions += 1
        else:
            wrong_first_positions += 1

    return correct_first_positions / (correct_first_positions + wrong_first_positions)


def get_ltr_accuracy_and_make_report(result, formatted_time, file_name, folder='learn_to_rank'):
    result_report = ["\n"]

    kendall_tau_scores = evaluate_kendall_tau(result)
    average_tau = calculate_average_score(kendall_tau_scores)
    average_tau_text = "Kendall Tau értékek átlaga: " + str(average_tau)
    result_report.append(average_tau_text)

    rho = evaluate_spearman_rho(result)
    average_rho = calculate_average_score(rho)
    average_rho_text = "Spearman's rho értékek átlaga: " + str(average_rho)
    result_report.append(average_rho_text)

    predicted_first_positions = count_first_position_predictions(result)
    predicted_first_positions_text = "Eltalált elsőhelyezettek: " + str(predicted_first_positions)
    result_report.append(predicted_first_positions_text)

    result_report.append("\n")

    for index, (group, data) in enumerate(result.items()):
        if index <= 10:
            race_report = (
                'group_id: ' + str(group) + '\n' +
                'predicted_positions: ' + ', '.join(map(str, data['predicted_positions'])) + '\n' +
                'original_RES21: ' + ', '.join(map(str, data['original_RES21'])) + '\n'
            )

            result_report.append(race_report)
        else:
            break

    for item in result_report:
        print(item)

    file_name = create_text_file(formatted_time, folder, file_name)
    write_to_file(result_report, file_name)


def pairwise_learn_to_rank_pairwise(df, target, formatted_time):
    result = start_ranking(df, 'pairwise')
    get_ltr_accuracy_and_make_report(result, formatted_time, 'pairwise_learn_to_rank_pairwise')

    return None


def pairwise_learn_to_rank_ndcg(df, target, formatted_time):
    result = start_ranking(df, 'ndcg')
    get_ltr_accuracy_and_make_report(result, formatted_time, 'pairwise_learn_to_rank_ndcg')

    return None


def pairwise_learn_to_rank_map(df, target, formatted_time):
    result = start_ranking(df, 'map')
    get_ltr_accuracy_and_make_report(result, formatted_time, 'pairwise_learn_to_rank_map')

    return None
