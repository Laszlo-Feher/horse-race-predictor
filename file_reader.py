import os
import pandas as pd

from constants import *


def check_file_existence(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)

    if os.path.isfile(file_path):
        return file_name
    else:
        return None


def get_result_filename(folder_path, file_name):
    res_file_name = (
            '20' + file_name[10] + file_name[11] +  # year 2018
            file_name[4] + file_name[5] +  # month 09
            file_name[6] + file_name[7] +  # day 10
            '_CH_DAY_' +  # fix string
            file_name[1] + file_name[2] + file_name[3] +  # track code ALB
            '.txt')  # file type

    checked_file_name = check_file_existence(folder_path, res_file_name)
    return checked_file_name


# get_file_names
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
            res_files.append(get_result_filename(FILE_PATH_RES, file_name))

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
