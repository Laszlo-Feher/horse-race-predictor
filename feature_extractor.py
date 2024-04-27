from debug import print_dataframe
from file_reader import *
import pandas as pd


def remove_columns(dataframe, columns_to_remove):
    new_dataframe = dataframe.copy()

    new_dataframe = new_dataframe.drop(columns=columns_to_remove, errors='ignore')

    return new_dataframe


string_to_number_mapping = {
    HOR_FIELD_NAMES[5]: {
        '2': 1,
        '2UP': 2,
        '3': 3,
        '3UP': 4,
        '4': 5,
        '4UP': 6,
        '5': 7,
        '5UP': 8
    },
    RAC_FIELD_NAMES[9]: {
        '2': 1,
        '2UP': 2,
        '3': 3,
        '3UP': 4,
        '4': 5,
        '4UP': 6,
        '5': 7,
        '5UP': 8
    },
    ENT_FIELD_NAMES[11]: {
        'f': 1,
        'm': 2,
        'g': 3,
        'c': 4
    }
}


def replace_strings_with_numbers(df, column_name):
    # Define a mapping of string values to numeric values

    # Use the map function to replace strings with numbers
    df[column_name] = df[column_name].map(string_to_number_mapping[column_name])

    return df


def group_races_by_id(df, race_id):
    df.reset_index(drop=True, inplace=True)

    df['ID'] = 0

    for index, row in df.iterrows():
        if index > 0 and row['RES4'] > df.at[index - 1, 'RES4']:
            race_id = race_id + 1
        df.at[index, 'ID'] = race_id

    return df, race_id


def remove_rows_with_zero_res(df):
    return df[df[RES_TARGET] != '0']


def convert_result_type(df, convert_to_binary):
    if convert_to_binary:
        for index, row in df.iterrows():
            if row[RES_TARGET] != 1:
                df.at[index, RES_TARGET] = 0
            else:
                df.at[index, RES_TARGET] = 1
    else:
        df[RES_TARGET] = df[RES_TARGET].astype(int)

    return df


def convert_raw_to_extracted_data(r_raw_data, e_raw_data, h_raw_data, res_raw_data, convert_to_binary, race_id):

    r_selected_df = pd.DataFrame(columns=RAC_FIELD_NAMES)
    h_selected_df = pd.DataFrame(columns=HOR_FIELD_NAMES)

    for index, row in e_raw_data.iterrows():

        id_to_match = row[ENT_ID_NAMES[0]]

        if id_to_match in r_raw_data[RAC_ID_NAMES[0]].values:

            matching_row = r_raw_data[r_raw_data[RAC_ID_NAMES[0]] == id_to_match]

            if r_selected_df.empty:
                r_selected_df = matching_row
            else:
                r_selected_df = pd.concat([r_selected_df, matching_row], axis=0, ignore_index=True)

    for index, row in e_raw_data.iterrows():

        id_to_match = row[ENT_ID_NAMES[0]]
        id_to_match2 = row[ENT_ID_NAMES[1]]

        if id_to_match in h_raw_data[HOR_ID_NAMES[0]].values and id_to_match2 in h_raw_data[HOR_ID_NAMES[1]].values:
            matching_row = h_raw_data[(h_raw_data[HOR_ID_NAMES[0]] == id_to_match) & (h_raw_data[HOR_ID_NAMES[1]].values == id_to_match2)].iloc[0]
            matching_row = pd.DataFrame(matching_row).T

            if h_selected_df.empty:
                h_selected_df = matching_row
            else:
                h_selected_df = pd.concat([h_selected_df, matching_row], axis=0, ignore_index=True)

    result_dataframe = pd.concat([e_raw_data, r_selected_df, h_selected_df, res_raw_data], axis=1)

    # result_dataframe = remove_columns(result_dataframe, HOR_ID_NAMES + ENT_ID_NAMES + RAC_ID_NAMES + RES_ID_NAMES + ['E26', 'E26', 'E28', 'E34', 'R15', 'H18', 'H20', 'H41'])

    result_dataframe = replace_strings_with_numbers(result_dataframe, HOR_FIELD_NAMES[5])
    result_dataframe = replace_strings_with_numbers(result_dataframe, RAC_FIELD_NAMES[9])
    result_dataframe = replace_strings_with_numbers(result_dataframe, ENT_FIELD_NAMES[11])

    # check if there are missing rows
    if len(r_selected_df.index) == len(e_raw_data.index) == len(h_selected_df.index):

        result_dataframe = remove_rows_with_zero_res(result_dataframe)

        result_dataframe[RES_TARGET] = result_dataframe[RES_TARGET].astype(int)

        result_dataframe, race_id = group_races_by_id(result_dataframe, race_id)

        result_dataframe = convert_result_type(result_dataframe, convert_to_binary)

        # search for others:
        result_dataframe = remove_columns(result_dataframe, HOR_ID_NAMES + ENT_ID_NAMES + RAC_ID_NAMES + RES_ID_NAMES + ['E26', 'E34', 'H20', 'H41'])

        return result_dataframe, race_id
    else:
        return None, race_id


def check_same_file_name(r_files, e_files, h_files):
    for r_file_name, e_file_name, h_file_name in zip(r_files, e_files, h_files):
        if r_file_name[:9] + r_file_name[10:] != e_file_name[:9] + e_file_name[10:] != h_file_name[:9] + h_file_name[10:]:
            return False

    return True


# TODO iterator kathelyezese az if-ek miatt
def extract_and_format_data(amount_of_files, is_divided_to_races=False, convert_to_binary=True):
    r_files, e_files, h_files, res_files = get_file_paths(amount_of_files)

    if not check_same_file_name(r_files, e_files, h_files):
        return None

    r_raw_data, e_raw_data, h_raw_data, res_raw_data = None, None, None, None

    feature_vectors = None

    iterator = 0
    calculated_files = 0
    race_id = 0
    for r_file_name, e_file_name, h_file_name, res_file_name in zip(r_files, e_files, h_files, res_files):
        if None in (r_file_name, e_file_name, h_file_name, res_file_name):
            iterator += 1
            percent = iterator / amount_of_files
            if iterator % 5 == 0:
                print("Completed: " + str(int(percent * 100)) + "/100%")
            continue
        r_dataframe = read_file(FILE_PATH_FILTERED_DATA, r_file_name, RAC_ID + RAC_FIELDS, 'csv')
        e_dataframe = read_file(FILE_PATH_FILTERED_DATA, e_file_name, ENT_ID + ENT_FIELDS, 'csv')
        h_dataframe = read_file(FILE_PATH_FILTERED_DATA, h_file_name, HOR_ID + HOR_FIELDS, 'csv')
        res_dataframe = read_file(FILE_PATH_FILTERED_RES, res_file_name, RES_ID + RES_FIELDS, 'txt')

        if r_dataframe is not None and e_dataframe is not None and h_dataframe is not None and res_dataframe is not None:
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

            # TODO create field names

            r_raw_data.columns = RAC_FIELD_NAMES
            e_raw_data.columns = ENT_FIELD_NAMES
            h_raw_data.columns = HOR_FIELD_NAMES
            res_raw_data.columns = RES_FIELD_NAMES

            extracted_data, race_id = convert_raw_to_extracted_data(r_raw_data, e_raw_data, h_raw_data, res_raw_data, convert_to_binary, race_id)
            r_raw_data, e_raw_data, h_raw_data, res_raw_data = None, None, None, None

            if extracted_data is not None:
                calculated_files += 1
                if not is_divided_to_races:
                    if feature_vectors is None:
                        feature_vectors = extracted_data
                    else:
                        feature_vectors = pd.concat([feature_vectors, extracted_data], axis=0)
                else:
                    if feature_vectors is None:
                        feature_vectors = [extracted_data]
                    else:
                        feature_vectors.append(extracted_data)

        iterator += 1
        percent = iterator / amount_of_files
        if iterator % 5 == 0:
            print("Completed: " + str(int(percent * 100)) + "/100%")

    print("Calculated files/ all files:  " + str(calculated_files) + "/" + str(iterator))
    print("Number of races: " + str(race_id+1))

    return feature_vectors
