import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def keep_same_number_of_zeros_as_ones(df):
    ones = df[df['RES21'] == 1]
    zeros = df[df['RES21'] == 0].head(len(ones))

    result_df = pd.concat([ones, zeros])

    return result_df


def learn_and_test(df, target):
    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

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

    # print(x_predicted_ps, y_test)

    ones_all = 0
    ones_true = 0
    ones_false = 0
    zeros_all = 0
    zeros_true = 0
    zeros_false = 0
    for i, j in zip(y_test, x_predicted_ps):
        if i == 1:
            ones_all += 1
            if i == j:
                ones_true += 1
            else:
                ones_false += 1
        else:
            zeros_all += 1
            if i == j:
                zeros_true += 1
            else:
                zeros_false += 1

    print("ones: " + str(ones_true/ones_all))
    print("zeros: " + str(zeros_true/zeros_all))

    return result
