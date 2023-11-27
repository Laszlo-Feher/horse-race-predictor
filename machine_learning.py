import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def keep_same_number_of_zeros_as_ones(df):
    ones = df[df['RES21'] == 1]
    zeros = df[df['RES21'] == 0].head(len(ones))

    result_df = pd.concat([ones, zeros])

    return result_df


def get_zero_and_one_accuracy(y_data, x_data):
    ones_all = 0
    ones_true = 0
    zeros_all = 0
    zeros_true = 0
    for i, j in zip(y_data, x_data):
        # print("i, j = ", i, j)
        if i == 1:
            ones_all += 1
            if i == j:
                ones_true += 1
        else:
            zeros_all += 1
            if i == j:
                zeros_true += 1

    print("ones: " + str(ones_true/ones_all))
    print("zeros: " + str(zeros_true/zeros_all))


def classification_with_bulk_fvs(df, target):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    df = df.fillna(df.median())
    x_train, x_test, y_train, y_test = train_test_split(df.drop(target, axis='columns'), df[target], test_size=0.2)
    model = SVC()
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    return result


def classification_with_equal_results(df, target):
    df = keep_same_number_of_zeros_as_ones(df)
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


def split_array(array, percentage):
    # Calculate the index for the split
    split_index = int(len(array) * percentage)

    # Split the array
    part1 = array[:split_index]
    part2 = array[split_index:]

    return part1, part2


def classification_with_individual_results(df, target):
    # df = df.fillna(df.median())
    df = [d.fillna(d.median()) for d in df]
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

    # x_predicted = model.predict(x_test)
    # x_predicted_ps = pd.Series(x_predicted)

    # get_zero_and_one_accuracy(y_test, x_predicted_ps)

    return result


def learn_and_test(df, target):
    result = classification_with_equal_results(df, target)
    # print(df)
    return result
