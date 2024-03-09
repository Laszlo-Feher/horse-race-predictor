import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from constants import RES_TARGET


def keep_same_number_of_zeros_as_ones(df, is_divided_to_races):
    result_df = None

    if not is_divided_to_races:
        ones = df[df[RES_TARGET] == 1]
        zeros = df[df[RES_TARGET] == 0].head(len(ones))

        result_df = pd.concat([ones, zeros])
    else:
        print("not implemented")
        # if result_df is None:
        #     result_df = [extracted_data]
        # else:
        #     result_df.append(extracted_data)

    ones = df[df[RES_TARGET] == 1]
    zeros = df[df[RES_TARGET] == 0].head(len(ones))

    result_df = pd.concat([ones, zeros])

    return result_df


def get_zero_and_one_accuracy(y_data, x_data):
    result_ones_all = 0
    result_ones_true = 0
    result_zeros_all = 0
    result_zeros_true = 0
    predicted_ones_all = 0
    predicted_ones_true = 0
    predicted_zeros_all = 0
    predicted_zeros_true = 0
    for i, j in zip(y_data, x_data):
        # print("i, j = ", i, j)
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

    # print("ones: " + str(result_ones_true/result_ones_all))
    # print("zeros: " + str(result_zeros_true/result_zeros_all))
    print("\n")
    print("Predikált 1-esek: " + str(predicted_ones_all))
    print("Ebből helyes: " + str(predicted_ones_true))
    print("Predikált 1-esek pontossága: " + str(predicted_ones_true/predicted_ones_all))
    print("Valós 1-esek: " + str(result_ones_all))
    print("Valós 1-esek pontossága: " + str(result_ones_true/result_ones_all))

    print("\n")
    print("Predikált 0-ások: " + str(predicted_zeros_all))
    print("Ebből helyes: " + str(predicted_zeros_true))
    print("Predikált 0-ások pontossága: " + str(predicted_zeros_true/predicted_zeros_all))
    print("Valós 0-ások: " + str(result_zeros_all))
    print("Valós 0-ások pontossága: " + str(result_ones_true/result_zeros_all))


# TODO hiba fileonkent nem racenkent van
# becsles manualis megnezese

# TODO tanitas balance, test osszes
# 1,2,3 cimke 1-es és 4+ 0-as
# utana talan elso 3-bol 1-es
# csak a valosag és a predikcio 1-es
#
#
#
#


def get_accuracy_for_values(y_data, x_data, max_value):
    for value in range(1, max_value + 1):
        correct_predictions = sum(1 for i, j in zip(y_data, x_data) if i == value and i == j)
        total_occurrences = sum(1 for i in y_data if i == value)

        print("Accuracy for {}: {:.2%}".format(value, correct_predictions / total_occurrences) if total_occurrences > 0 else "No occurrences of {} in the data".format(value))


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

    x_predicted = model.predict(x_test)
    x_predicted_ps = pd.Series(x_predicted)

    get_zero_and_one_accuracy(y_test, x_predicted_ps)

    return result


def classification_with_equal_results(df_original, target, is_divided_to_races):
    df = keep_same_number_of_zeros_as_ones(df_original, is_divided_to_races)
    df = df.fillna(df.median())

    x_train, x_test, y_train, y_test = train_test_split(df.drop(target, axis='columns'), df[target], test_size=0.2)
    # x_train = df.drop(target, axis='columns')
    # y_train = df[target]
    #
    # x_test = df_original.drop(target, axis='columns')
    # y_test = df_original[target]

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


#teszt kiegyenlitett, fit rendes
#racekent mennyi 1-es
#fileba kiadjaa futomonkénti sztatisztika
#10 verseny random fogadsz/ez a teszt osszehasonlitas
#vegen ennyit nyerhetsz osszehasonlitas

#learn to rank

def classification_with_individual_results(df, target):
    # df = df.fillna(df.median())
    #eldobni inkabb, megnezni, mennyit dob el
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

    x_predicted = model.predict(x_test)
    x_predicted_ps = pd.Series(x_predicted)

    get_zero_and_one_accuracy(y_test, x_predicted_ps)

    return result


def random_forest_classifier(df, target):
    # df = df.fillna(df.median())
    df = [d.fillna(d.median()) for d in df]
    df_train, df_test = split_array(df, 0.8)
    x_train, y_train, x_test, y_test = None, None, None, None
    result = []
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("cica",len(df), len(df_train), len(df_test))
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

        get_accuracy_for_values(y_test, x_predicted_ps, 14)

        score = model.score(x_test, y_test)
        result.append(score)
    print(len(df_test))
    # x_predicted = model.predict(x_test)
    # x_predicted_ps = pd.Series(x_predicted)
    #
    # get_accuracy_for_values(y_test, x_predicted_ps, 14)

    return result


def learn_and_test(df, target, algorythm, is_divided_to_races):
    if algorythm == "random_forest_classifier":
        result = random_forest_classifier(df, target)
    # elif algorythm == "classification":
    #     result = classification_with_equal_results(df, target)
    else:
        result = classification_with_equal_results(df, target, is_divided_to_races)
    # result = classification_with_individual_results(df, target)
    # print(df)
    return result
