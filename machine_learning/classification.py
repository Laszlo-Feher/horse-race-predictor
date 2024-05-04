import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from io_utils.file_writer import *
from feature_vectors.constants import RES_TARGET


def custom_group_split(df, id_column, test_size=0.2, random_state=None):
    """
    Splits the DataFrame into training and testing sets while preserving groups (IDs).

    Parameters:
        df (DataFrame): The DataFrame to be split.
        id_column (str): The name of the column containing the IDs.
        test_size (float or int, optional): The proportion of the dataset to include in the test split.
                                             Default is 0.2.
        random_state (int, RandomState instance or None, optional): Controls the randomness of the split.
                                                                    Default is None.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """
    # Extract the group (ID) column from the DataFrame
    groups = df[id_column]

    # Initialize GroupShuffleSplit with specified parameters
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Perform the data split while preserving groups (IDs)
    for train_index, test_index in gss.split(df, groups=groups):
        # Obtain the training data using the training indices
        train_data = df.iloc[train_index]
        # Obtain the testing data using the testing indices
        test_data = df.iloc[test_index]

    # Return the training and testing DataFrames
    return train_data, test_data


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


def keep_same_number_of_zeros_as_ones(df, is_divided_to_races=False):
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


def classify_by_race_without_conversion(df, target):

    df = df.fillna(df.median())
    x_train, x_test = custom_group_split(df, 'ID', 0.2, 42)

    y_train = x_train[target]
    x_train = x_train.drop(target, axis='columns')

    y_test = x_test[target]
    x_test = x_test.drop(target, axis='columns')

    model = SVC()
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)

    x_predicted = model.predict(x_test)
    x_predicted_ps = pd.Series(x_predicted)

    get_zero_and_one_accuracy(y_test, x_predicted_ps)

    return result


def split_to_first_3_and_the_rest(df, target):
    df = df.fillna(df.median())
    x_train, x_test = custom_group_split(df, 'ID', 0.2, 42)

    # Set target column to 1 for IDs 1 to 3
    x_train.loc[df[target].isin([1, 2, 3]), target] = 1
    # Set target column to 0 for IDs not in 1 to 3
    x_train.loc[~df[target].isin([1, 2, 3]), target] = 0

    x_train = keep_same_number_of_zeros_as_ones(x_train)

    y_train_top_3_binary = x_train[target]
    x_train = x_train.drop(target, axis='columns')

    temp_x_test = x_test.copy()
    temp_x_test.loc[df[target].isin([1]), target] = 1
    temp_x_test.loc[~df[target].isin([1]), target] = 0
    y_test_top_1_binary = temp_x_test[target]

    x_test.loc[df[target].isin([1, 2, 3]), target] = 1
    x_test.loc[~df[target].isin([1, 2, 3]), target] = 0
    y_test_top_3_binary = x_test[target]
    x_test = x_test.drop(target, axis='columns')

    # predict top 3
    model_top_3 = SVC()
    y_train_top_3_binary = y_train_top_3_binary.astype(int)
    y_test_top_3_binary = y_test_top_3_binary.astype(int)

    # Print the number of 0 and 1
    # value_counts = y_train_top_3_binary.value_counts()
    # count_0 = value_counts.get(0, 0)
    # count_1 = value_counts.get(1, 0)
    # print("Count of 0:", count_0)
    # print("Count of 1:", count_1)

    model_top_3.fit(x_train, y_train_top_3_binary)
    result = model_top_3.score(x_test, y_test_top_3_binary)

    x_predicted = model_top_3.predict(x_test)
    x_predicted_ps = pd.Series(x_predicted)

    get_zero_and_one_accuracy(y_test_top_3_binary, x_predicted_ps)

    filtered_features = []
    filtered_results = []

    for (index, x_test_row), y_test_top_1_binary_row, x_predicted_row in zip(x_test.iterrows(), y_test_top_1_binary, x_predicted):

        if x_predicted_row == 1:
            filtered_features.append(x_test_row)
            filtered_results.append(y_test_top_1_binary_row)

    # Convert the list of rows to a DataFrame
    filtered_features_df = pd.DataFrame(filtered_features, columns=x_test.columns)
    filtered_features_df.reset_index(drop=True, inplace=True)
    filtered_results_df = pd.DataFrame(filtered_results)

    filtered_features_df[target] = filtered_results_df

    # predict top 1
    model_top_1 = SVC()

    x_train_top_1, x_test_top_1 = custom_group_split(df, 'ID', 0.2, 42)
    y_train_top_1 = x_train_top_1[target]
    y_test_top_1 = x_test_top_1[target]

    model_top_1.fit(x_train_top_1, y_train_top_1)
    result = model_top_1.score(x_test_top_1, y_test_top_1)

    x_predicted_top_1 = model_top_1.predict(x_test_top_1)
    x_predicted_ps_top_1 = pd.Series(x_predicted_top_1)

    get_zero_and_one_accuracy(y_test_top_1, x_predicted_ps_top_1)

    return result,
