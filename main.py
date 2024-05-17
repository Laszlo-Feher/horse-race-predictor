import datetime
import time

from feature_vectors.feature_extractor import *
from io_utils.file_filterer import filter_files_and_copy
from io_utils.file_writer import write_dataframe_to_file
from machine_learning.machine_learning import *
from utils.debug import *

# all files: 6589
# read files: 6475
# error - indices exceed the number of columns: TODO
# number of races: 47081
# estimated time: 994.45 in seconds

# number of rows: 352556
# number of rows having nan values: 7854
# number of rows after nan is removed: 305200

# For value 14, the length of the array is 47
# For value 13, the length of the array is 62
# For value 12, the length of the array is 937
# For value 11, the length of the array is 1174
# For value 10, the length of the array is 3017
# For value 9, the length of the array is 4613
# For value 8, the length of the array is 6999
# For value 7, the length of the array is 9148
# For value 6, the length of the array is 8994
# For value 5, the length of the array is 4282
# For value 4, the length of the array is 1217
# For value 3, the length of the array is 409
# For value 2, the length of the array is 267
# For value 1, the length of the array is 179

def main(copy_files, create_feature_vectors, run_learning, amounts_of_files, amounts_of_races, algorythm, median):
    if copy_files:
        filter_files_and_copy()

    if create_feature_vectors:
        feature_vectors = extract_and_format_data(amounts_of_files)
        feature_vectors = reset_dataframe_index(feature_vectors)
        feature_vectors = feature_vectors.astype(float)

        write_dataframe_to_file(feature_vectors, FILE_PATH_EXPORTED_DATAFRAME)

        exported_feature_vectors = read_dataframe_from_file(FILE_PATH_EXPORTED_DATAFRAME)

        print(check_dataframe_equality(exported_feature_vectors, feature_vectors))

    if run_learning:
        for amount_of_races in amounts_of_races:
            print("Start preparing feature vectors...")
            start = time.time()

            exported_feature_vectors = read_dataframe_from_file(FILE_PATH_EXPORTED_DATAFRAME)
            selected_feature_vectors = get_rows_by_id_range(exported_feature_vectors, amount_of_races)

            num_ids = get_num_unique_ids(selected_feature_vectors)
            end = time.time()

            print("Feature vectors prepared!")
            print("Time Usage: " + str(round((end - start), 2)) + " in seconds")
            print("Number of unique IDs:", num_ids, "\n")

            # check selected_feature_vectors, no ml
            # print(selected_feature_vectors)
            # selected_feature_vectors = None

            # check dataframe before learning
            if check_dataframe(selected_feature_vectors):
                print("DataFrame is valid!")
                print("Start machine learning...")

                start = time.time()
                if median == "drop":
                    rows_with_nan_ids = selected_feature_vectors.loc[selected_feature_vectors.isnull().any(axis=1), 'ID'].tolist()
                    selected_feature_vectors = selected_feature_vectors[~selected_feature_vectors['ID'].isin(rows_with_nan_ids)]

                    # remove only nan values, and races where only 1 value or doesn't have 1 value
                    # selected_feature_vectors = selected_feature_vectors.dropna()
                    races = selected_feature_vectors.groupby('ID')
                    for race_id, race_df in races:
                        if race_df['RES21'].min() != 1:
                            selected_feature_vectors = selected_feature_vectors[selected_feature_vectors['ID'] != race_id]
                        elif race_df['RES21'].max() == 1:
                            selected_feature_vectors = selected_feature_vectors[selected_feature_vectors['ID'] != race_id]

                    # selected_feature_vectors = get_race_participants(selected_feature_vectors)

                if median == "fill":
                    selected_feature_vectors = selected_feature_vectors.fillna(selected_feature_vectors.median())

                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%Y-%m-%d %H-%M-%S")
                formatted_time = formatted_time + " - " + str(amount_of_races)
                score = learn_and_test(selected_feature_vectors, RES_TARGET, algorythm, formatted_time)
                print('\n')
                end = time.time()
                print("Time Usage: " + str(round((end - start), 2)) + " in seconds")
                print('\n')
                # print("Test Accuracy: " + str(score))

    return 0


copy_files = False
create_feature_vectors = False
run_learning = True
amounts_of_files = 6589
amounts_of_races = [100, 200, 300, 500, 1000, 1500, 2000, 3000, 5000, 10000, 15000, 20000]
algorythm = "all"
median = "drop"

main(copy_files, create_feature_vectors, run_learning, amounts_of_files, amounts_of_races, algorythm, median)
