import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
from IPython.display import clear_output

FILE_PATH_DATA = "../Post Time Daily Files/Exported Data"
FILE_PATH_RES = "../Post Time Daily Files/Result Charts"

RAC_ID = 4
ENT_ID = 3
HOR_ID = 3
RES_ID = 4

HOR_FIELDS = [14, 15, 16, 18, 19, 20, 22, 24, 27, 28, 29, 30, 41, 48]
ENT_FIELDS = [9, 10, 11, 12, 13, 14, 15, 21, 26, 28, 34, 40, 84, 45]
RAC_FIELDS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 25, 28]
RES_FIELDS = [1, 4, 21]

HOR_FIELD_NAMES = ['H3', 'H14', 'H15', 'H16', 'H18', 'H19', 'H20', 'H22', 'H24', 'H27', 'H28', 'H29', 'H30', 'H41', 'H48']
ENT_FIELD_NAMES = ['E3', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E21', 'E26', 'E28', 'E34', 'E40', 'E84', 'E45']
RAC_FIELD_NAMES = ['R4', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R18', 'R19', 'R25', 'R28']
RES_FIELD_NAMES = ['RES0', 'RES4', 'RES21']


def get_field_names(letter):
    field_names = []
    for field in RAC_FIELDS:
        field_names.append(letter + str(field))
    print(field_names)


def check_file_existence(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)

    if os.path.isfile(file_path):
        return file_name
    else:
        return None


def get_result_path(folder_path, file_name):
    res_file_name = (
            '20' + file_name[10] + file_name[11] +  # year 2018
            file_name[4] + file_name[5] +  # month 09
            file_name[6] + file_name[7] +  # day 10
            '_CH_DAY_' +  # fix string
            file_name[1] + file_name[2] + file_name[3] +  # track code ALB
            '.txt')  # file type

    checked_file_name = check_file_existence(folder_path, res_file_name)
    return checked_file_name


def get_file_paths(amount_of_files=0):
    r_files = []
    e_files = []
    h_files = []
    res_files = []

    file_names = os.listdir(FILE_PATH_DATA)
    file_names.sort()

    iterator = 0
    for file_name in file_names:
        if iterator >= amount_of_files * 5 and amount_of_files != 0:
            break

        if file_name[9] == 'R':
            r_files.append(file_name)
            res_files.append(get_result_path(FILE_PATH_RES, file_name))

        if file_name[9] == 'E':
            e_files.append(file_name)

        if file_name[9] == 'H':
            h_files.append(file_name)

        iterator += 1

    return r_files, e_files, h_files, res_files


def adjust_field_numbers(selected_fields):
    return [field_number - 1 for field_number in selected_fields]


def read_file(file_path, file_name, selected_fields, file_extension='csv'):
    selected_fields_adjusted = adjust_field_numbers(selected_fields)

    if file_extension.lower() == 'csv':
        data = pd.read_csv(file_path + '/' + file_name, sep=',', header=None)
    elif file_extension.lower() == 'txt':
        data = []
        with open(file_path + '/' + file_name, 'r') as file:
            for line in file:
                if line.startswith('"H"'):
                    data.append(line.split(','))

        data = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file extension. Only 'csv' and 'txt' are supported.")

    selected_data = data.iloc[:, selected_fields_adjusted]

    return selected_data


def remove_columns(dataframe, columns_to_remove):

    new_dataframe = dataframe.copy()

    new_dataframe = new_dataframe.drop(columns=columns_to_remove, errors='ignore')

    return new_dataframe


def convert_raw_to_extracted_data(r_raw_data, e_raw_data, h_raw_data, res_raw_data):
    extracted_data = []

    for index, row in res_raw_data.iterrows():
        if row['RES21'] != '1':
            res_raw_data.at[index, 'RES21'] = 0
        else:
            res_raw_data.at[index, 'RES21'] = 1

    result_dataframe = pd.DataFrame(columns=ENT_FIELD_NAMES + RAC_FIELD_NAMES + HOR_FIELD_NAMES + ['target'])

    r_selected_df = pd.DataFrame(columns=RAC_FIELD_NAMES)
    h_selected_df = pd.DataFrame(columns=HOR_FIELD_NAMES)

    for index, row in e_raw_data.iterrows():

        id_to_match = row['E3']

        if id_to_match in r_raw_data['R4'].values:

            matching_row = r_raw_data[r_raw_data['R4'] == id_to_match]

            if r_selected_df.empty:
                r_selected_df = matching_row
            else:
                r_selected_df = pd.concat([r_selected_df, matching_row], axis=0, ignore_index=True)

    for index, row in e_raw_data.iterrows():

        id_to_match = row['E3']

        if id_to_match in h_raw_data['H3'].values:

            matching_row = h_raw_data[h_raw_data['H3'] == id_to_match].iloc[0]
            matching_row = pd.DataFrame(matching_row).T

            if h_selected_df.empty:
                h_selected_df = matching_row
            else:
                h_selected_df = pd.concat([h_selected_df, matching_row], axis=0, ignore_index=True)

    result_dataframe = pd.concat([e_raw_data, r_selected_df, h_selected_df, res_raw_data], axis=1)
    result_dataframe = remove_columns(result_dataframe, ['H3', 'E3', 'R4', 'RES0', 'RES4', 'E26', 'E26', 'E28', 'E34', 'R15', 'H18', 'H20', 'H41'])

    return result_dataframe


def learn_and_test(df, target):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    df = df.fillna(df.median())
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis='columns'), df[target], test_size=0.2)
    model = SVC()
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    model.fit(X_train, y_train)
    result = model.score(X_test, y_test)
    return result


def extract_and_format_data(amount_of_files):
    r_files, e_files, h_files, res_files = get_file_paths(amount_of_files)
    r_raw_data, e_raw_data, h_raw_data, res_raw_data = None, None, None, None

    feature_vectors = None
    print('\n' * 80)

    iterator = 0
    for r_fileName, e_fileName, h_fileName, res_fileName in zip(r_files, e_files, h_files, res_files):
        if None in (r_fileName, e_fileName, h_fileName, res_fileName):
            iterator += 1
            percent = iterator / amount_of_files
            print('\n' * 80)
            print("Completed: " + str(int(percent * 100)) + "/100%")
            continue

        r_dataframe = read_file(FILE_PATH_DATA, r_fileName, [RAC_ID] + RAC_FIELDS, 'csv')
        e_dataframe = read_file(FILE_PATH_DATA, e_fileName, [ENT_ID] + ENT_FIELDS, 'csv')
        h_dataframe = read_file(FILE_PATH_DATA, h_fileName, [HOR_ID] + HOR_FIELDS, 'csv')
        res_dataframe = read_file(FILE_PATH_RES, res_fileName, RES_FIELDS, 'txt')

        if r_raw_data is None:
            r_raw_data = r_dataframe
        else:
            r_raw_data = pd.concat([r_raw_data, r_dataframe], axis=0)

        if e_raw_data is None:
            e_raw_data = e_dataframe
        else:
            e_raw_data = pd.concat([e_raw_data, e_dataframe], axis=0)

        if h_raw_data is None:
            h_raw_data = h_dataframe
        else:
            h_raw_data = pd.concat([h_raw_data, h_dataframe], axis=0)

        if res_raw_data is None:
            res_raw_data = res_dataframe
        else:
            res_raw_data = pd.concat([res_raw_data, res_raw_data], axis=0)

        if r_raw_data is None or e_raw_data is None or h_raw_data is None or res_raw_data is None:
            print("1 or more file(s) is/are missing!")
            return 0

        r_raw_data = r_raw_data.reset_index(drop=True)
        e_raw_data = e_raw_data.reset_index(drop=True)
        h_raw_data = h_raw_data.reset_index(drop=True)
        res_raw_data = res_raw_data.reset_index(drop=True)

        r_raw_data.columns = RAC_FIELD_NAMES
        e_raw_data.columns = ENT_FIELD_NAMES
        h_raw_data.columns = HOR_FIELD_NAMES
        res_raw_data.columns = RES_FIELD_NAMES

        extracted_data = convert_raw_to_extracted_data(r_raw_data, e_raw_data, h_raw_data, res_raw_data)
        r_raw_data, e_raw_data, h_raw_data, res_raw_data = None, None, None, None

        if feature_vectors is None:
            feature_vectors = extracted_data
        else:
            feature_vectors = pd.concat([feature_vectors, extracted_data], axis=0)

        iterator += 1
        percent = iterator/amount_of_files
        print('\n' * 80)
        print("Completed: " + str(int(percent * 100)) + "/100%")

    return feature_vectors


def main():
    start = time.time()
    feature_vectors = extract_and_format_data(145)
    score = learn_and_test(feature_vectors, 'RES21')
    print('\n')
    end = time.time()
    print("Time Usage: " + str(round((end - start), 2)) + " in seconds")
    print('\n')
    print("Test Accuracy: " + str(score))
    return 0

main()

