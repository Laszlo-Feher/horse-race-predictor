import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from io_utils.file_writer import *
from feature_vectors.constants import RES_TARGET


def convert_dataframe_to_array_by_id(df):
    """
    Groups DataFrame rows by the 'ID' column and returns a list of DataFrames
    where each DataFrame contains rows with the same ID.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    list: List of DataFrames where each DataFrame contains rows with the same ID.
    """
    grouped_data = []
    for _, row in df.iterrows():
        id_value = row['ID']
        found = False
        for group_df in grouped_data:
            if id_value in group_df['ID'].values:
                group_df = pd.concat([group_df, row], axis=0)
                found = True
                break
        if not found:
            grouped_data.append(df[df['ID'] == id_value].copy())
    return grouped_data


def convert_result_to_binary(df):
    for index, row in df.iterrows():
        if row[RES_TARGET] != 1:
            df.at[index, RES_TARGET] = 0
        else:
            df.at[index, RES_TARGET] = 1

    return df


def split_array(array, percentage):
    # Calculate the index for the split
    split_index = int(len(array) * percentage)

    # Split the array
    part1 = array[:split_index]
    part2 = array[split_index:]

    return part1, part2


def keep_same_number_of_zeros_as_ones(df, is_divided_to_races):
    result_df = None

    if not is_divided_to_races:
        ones = df[df[RES_TARGET] == 1]
        zeros = df[df[RES_TARGET] == 0].head(len(ones))

        result_df = pd.concat([ones, zeros])
    else:
        print("not implemented")

    return result_df


def get_zero_and_one_accuracy(y_data, x_data):
    result_report = []
    result_ones_all = 0
    result_ones_true = 0
    result_zeros_all = 0
    result_zeros_true = 0
    predicted_ones_all = 0
    predicted_ones_true = 0
    predicted_zeros_all = 0
    predicted_zeros_true = 0
    for i, j in zip(y_data, x_data):
        if i == 1:
            result_ones_all += 1
            if i == j:
                result_ones_true += 1
        else:
            result_zeros_all += 1
            if i == j:
                result_zeros_true += 1

        if j == 1:
            predicted_ones_all += 1
            if i == j:
                predicted_ones_true += 1
        else:
            predicted_zeros_all += 1
            if i == j:
                predicted_zeros_true += 1

    result_report.append("\n")

    predicted_ones = "Predikált 1-esek: " + str(predicted_ones_all)
    result_report.append(predicted_ones)

    predicted_ones_true_str = "Ebből helyes: " + str(predicted_ones_true)
    result_report.append(predicted_ones_true_str)

    if predicted_ones_all != 0:
        predicted_ones_accuracy_str = "Predikált 1-esek pontossága: " + str(predicted_ones_true / predicted_ones_all)
    else:
        predicted_ones_accuracy_str = "Predikált 1-esek pontossága: " + '0'
    result_report.append(predicted_ones_accuracy_str)

    real_ones = "Valós 1-esek: " + str(result_ones_all)
    result_report.append(real_ones)

    if result_ones_all != 0:
        real_ones_accuracy_str = "Valós 1-esek pontossága: " + str(result_ones_true / result_ones_all)
    else:
        real_ones_accuracy_str = "Valós 1-esek pontossága: " + '0'
    result_report.append(real_ones_accuracy_str)

    result_report.append("\n")

    predicted_zeros = "Predikált 0-ások: " + str(predicted_zeros_all)
    result_report.append(predicted_zeros)

    predicted_zeros_true_str = "Ebből helyes: " + str(predicted_zeros_true)
    result_report.append(predicted_zeros_true_str)

    if predicted_zeros_all != 0:
        predicted_zeros_accuracy_str = "Predikált 0-ások pontossága: " + str(predicted_zeros_true / predicted_zeros_all)
    else:
        predicted_zeros_accuracy_str = "Predikált 0-ások pontossága: " + '0'
    result_report.append(predicted_zeros_accuracy_str)

    real_zeros = "Valós 0-ások: " + str(result_zeros_all)
    result_report.append(real_zeros)

    if result_zeros_all != 0:
        real_zeros_accuracy_str = "Valós 0-ások pontossága: " + str(result_zeros_true / result_zeros_all)
    else:
        real_zeros_accuracy_str = "Valós 0-ások pontossága: " + '0'

    result_report.append(real_zeros_accuracy_str)

    for item in result_report:
        print(item)

    file_name = create_text_file()
    write_to_file(result_report, file_name)


def classification_with_bulk_fvs(df, target):
    df = convert_result_to_binary(df)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    df = df.fillna(df.median())
    x_train, x_test, y_train, y_test = train_test_split(df.drop(target, axis='columns'), df[target], test_size=0.2)
    model = SVC()
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)

    x_predicted = model.predict(x_test)
    x_predicted_ps = pd.Series(x_predicted)

    get_zero_and_one_accuracy(y_test, x_predicted_ps)

    return result


def classification_with_equal_results(df_original, target, is_divided_to_races=False):
    df = convert_result_to_binary(df_original)
    df = keep_same_number_of_zeros_as_ones(df_original, is_divided_to_races)
    df = df.fillna(df.median())

    x_train, x_test, y_train, y_test = train_test_split(df.drop(target, axis='columns'), df[target], test_size=0.2)

    model = SVC()
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)

    x_predicted = model.predict(x_test)
    x_predicted_ps = pd.Series(x_predicted)

    get_zero_and_one_accuracy(y_test, x_predicted_ps)

    return result


def classification_with_individual_results(df, target):
    df = convert_result_to_binary(df)
    df = df.fillna(df.median())
    df = convert_dataframe_to_array_by_id(df)

    df_train, df_test = split_array(df, 0.8)
    x_train, y_train, x_test, y_test = None, None, None, None
    result = []
    model = SVC()

    # x_train = df_train.drop(target, axis='columns')
    # y_train = df_train[target]

    # x_test = df_test.drop(target, axis='columns')
    # y_test = df_test[target]
    iterator = 0
    for element in df_train:
        element = element.astype(int)
        x_train = element.drop(target, axis='columns')
        y_train = element[target]
        model.fit(x_train, y_train)

        if iterator == 0:
            iterator = 1
            print(x_train, '\n', y_train)

    for element in df_test:
        element = element.astype(int)
        x_test = element.drop(target, axis='columns')
        y_test = element[target]

        x_predicted = model.predict(x_test)
        x_predicted_ps = pd.Series(x_predicted)

        get_zero_and_one_accuracy(y_test, x_predicted_ps)

        score = model.score(x_test, y_test)
        result.append(score)

    x_predicted = model.predict(x_test)
    x_predicted_ps = pd.Series(x_predicted)

    get_zero_and_one_accuracy(y_test, x_predicted_ps)

    return result


def split_to_first_3_and_the_rest(df, target):
    return "not implemented"

