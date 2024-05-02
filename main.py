import time

from feature_vectors.feature_extractor import *
from io_utils.file_filterer import filter_files_and_copy
from machine_learning.machine_learning import *
from utils.debug import *

# all files: 6589
# read files: 6475
# error - indices exceed the number of columns: TODO
# number of races: 47081
# estimated time: 994.45 in seconds


def main(copy_files, create_feature_vectors, run_learning, amount_of_races):
    if copy_files:
        filter_files_and_copy()

    if create_feature_vectors:
        feature_vectors = extract_and_format_data(6589)
        feature_vectors = reset_dataframe_index(feature_vectors)
        feature_vectors = feature_vectors.astype(float)

        write_dataframe_to_file(feature_vectors, FILE_PATH_EXPORTED_DATAFRAME)

        exported_feature_vectors = read_dataframe_from_file(FILE_PATH_EXPORTED_DATAFRAME)

        print(check_dataframe_equality(exported_feature_vectors, feature_vectors))

    if run_learning:
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

            score = learn_and_test(selected_feature_vectors, RES_TARGET, "pairwise_learn_to_rank", False)
            print('\n')
            end = time.time()
            print("Time Usage: " + str(round((end - start), 2)) + " in seconds")
            # print('\n')
            # print("Test Accuracy: " + str(score))

    return 0


copy_files = False
create_feature_vectors = False
run_learning = True
amount_of_races = 100

main(copy_files, create_feature_vectors, run_learning, amount_of_races)
