import os
import pandas as pd

FILE_PATH_DATA = "../Post Time Daily Files/Exported Data"
FILE_PATH_RES = "../Post Time Daily Files/Result Charts"

RAC_ID = 4
ENT_ID = 3
HOR_ID = 3

HOR_FIELDS = [14, 15, 16, 18, 19, 20, 22, 24, 27, 28, 29, 30, 41, 48]
ENT_FIELDS = [9, 10, 11, 12, 13, 14, 15, 21, 26, 28, 34, 40, 84, 45]
RAC_FIELDS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 25, 28]

HOR_FIELD_NAMES = ['H3', 'H14', 'H15', 'H16', 'H18', 'H19', 'H20', 'H22', 'H24', 'H27', 'H28', 'H29', 'H30', 'H41', 'H48']
ENT_FIELD_NAMES = ['E3', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E21', 'E26', 'E28', 'E34', 'E40', 'E84', 'E45']
RAC_FIELD_NAMES = ['R4', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R18', 'R19', 'R25', 'R28']


def get_field_names(letter):
    field_names = []
    for field in RAC_FIELDS:
        field_names.append(letter + str(field))
    print(field_names)


# get name instead
def get_result_file_name(file_name):
    res_file_name = (
            '20' + file_name[10] + file_name[11] +  # year 2018
            file_name[4] + file_name[5] +  # month 09
            file_name[6] + file_name[7] +  # day 10
            '_CH_DAY_' +  # fix string
            file_name[6] + file_name[7] + file_name[7] +  # track code ALB
            '.txt')  # file type

    return res_file_name


def get_file_paths(amount_of_files=0):
    r_files = []
    e_files = []
    h_files = []
    res_files = []

    iterator = 0
    for file_name in os.listdir(FILE_PATH_DATA):
        if iterator >= amount_of_files * 5 and amount_of_files != 0:
            break

        if file_name[9] == 'R':
            r_files.append(file_name)
            res_files.append(get_result_file_name(file_name))

        if file_name[9] == 'E':
            e_files.append(file_name)

        if file_name[9] == 'H':
            h_files.append(file_name)

        iterator += 1

    return r_files, e_files, h_files, res_files


def adjust_field_numbers(selected_fields):
    return [field_number - 1 for field_number in selected_fields]


def read_file(file_name, selected_fields):
    selected_fields_adjusted = adjust_field_numbers(selected_fields)

    data = pd.read_csv(FILE_PATH_DATA + '/' + file_name, sep=',', header=None)

    selected_data = data.iloc[:, selected_fields_adjusted]

    return selected_data


def convert_raw_to_extracted_data(r_raw_data, e_raw_data, h_raw_data):
    return 0


def extract_and_format_data(amount_of_files):
    r_files, e_files, h_files, res_files = get_file_paths(amount_of_files)
    r_raw_data, e_raw_data, h_raw_data = None, None, None

    for r_fileName, e_fileName, h_fileName in zip(r_files, e_files, h_files):
        r_dataframe = read_file(r_fileName, [RAC_ID] + RAC_FIELDS)
        e_dataframe = read_file(e_fileName, [ENT_ID] + ENT_FIELDS)
        h_dataframe = read_file(h_fileName, [HOR_ID] + HOR_FIELDS)

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

    r_raw_data = r_raw_data.reset_index(drop=True)
    e_raw_data = e_raw_data.reset_index(drop=True)
    h_raw_data = h_raw_data.reset_index(drop=True)

    r_raw_data.columns = RAC_FIELD_NAMES
    e_raw_data.columns = ENT_FIELD_NAMES
    h_raw_data.columns = HOR_FIELD_NAMES

    extracted_data = convert_raw_to_extracted_data(r_raw_data, e_raw_data, h_raw_data)

    return extracted_data


def main():
    feature_vectors = extract_and_format_data(1)
    print(feature_vectors)
    return 0


main()

