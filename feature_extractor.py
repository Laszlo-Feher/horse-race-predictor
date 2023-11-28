from file_reader import *
import pandas as pd


def remove_columns(dataframe, columns_to_remove):

    new_dataframe = dataframe.copy()

    new_dataframe = new_dataframe.drop(columns=columns_to_remove, errors='ignore')

    return new_dataframe


def convert_raw_to_extracted_data(r_raw_data, e_raw_data, h_raw_data, res_raw_data):

    for index, row in res_raw_data.iterrows():
        if row[RES_TARGET] != '1':
            res_raw_data.at[index, RES_TARGET] = 0
        else:
            res_raw_data.at[index, RES_TARGET] = 1

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
    result_dataframe = remove_columns(result_dataframe, HOR_ID_NAMES + ENT_ID_NAMES + RAC_ID_NAMES + ['RES0', 'RES4', 'E26', 'E26', 'E28', 'E34', 'R15', 'H18', 'H20', 'H41'])

    # check if there are missing rows
    if len(r_selected_df.index) == len(e_raw_data.index) == len(h_selected_df.index):
        return result_dataframe
    else:
        return None


def extract_and_format_data(amount_of_files, is_divided_to_races=False):
    r_files, e_files, h_files, res_files = get_file_paths(amount_of_files)
    r_raw_data, e_raw_data, h_raw_data, res_raw_data = None, None, None, None

    feature_vectors = None

    iterator = 0
    for r_fileName, e_fileName, h_fileName, res_fileName in zip(r_files, e_files, h_files, res_files):
        if None in (r_fileName, e_fileName, h_fileName, res_fileName):
            iterator += 1
            percent = iterator / amount_of_files
            print("Completed: " + str(int(percent * 100)) + "/100%")
            continue

        r_dataframe = read_file(FILE_PATH_DATA, r_fileName, RAC_ID + RAC_FIELDS, 'csv')
        e_dataframe = read_file(FILE_PATH_DATA, e_fileName, ENT_ID + ENT_FIELDS, 'csv')
        h_dataframe = read_file(FILE_PATH_DATA, h_fileName, HOR_ID + HOR_FIELDS, 'csv')
        res_dataframe = read_file(FILE_PATH_RES, res_fileName, RES_ID + RES_FIELDS, 'txt')

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

        #TODO create field names

        r_raw_data.columns = RAC_FIELD_NAMES
        e_raw_data.columns = ENT_FIELD_NAMES
        h_raw_data.columns = HOR_FIELD_NAMES
        res_raw_data.columns = RES_FIELD_NAMES

        extracted_data = convert_raw_to_extracted_data(r_raw_data, e_raw_data, h_raw_data, res_raw_data)
        r_raw_data, e_raw_data, h_raw_data, res_raw_data = None, None, None, None

        if extracted_data is not None:
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
        percent = iterator/amount_of_files
        print("Completed: " + str(int(percent * 100)) + "/100%")

    return feature_vectors
