import time

from feature_vectors.feature_extractor import *
from io_utils.file_filterer import filter_files_and_copy
from machine_learning.machine_learning import *


def main(copy_files, run_learning):
    if copy_files:
        filter_files_and_copy()

    if run_learning:
        start = time.time()
        feature_vectors = extract_and_format_data(1, False, False)
        if feature_vectors is None:
            print("Something went wrong during data extracting and formatting!")
            return 0

        # TODO: overview
        feature_vectors = feature_vectors.astype(float)
        score = learn_and_test(feature_vectors, RES_TARGET, "pairwise_learn_to_rank", False)
        print('\n')
        end = time.time()
        print("Time Usage: " + str(round((end - start), 2)) + " in seconds")
        print('\n')
        # print("Test Accuracy: " + str(score))

    return 0


copy_files = False
run_learning = True

main(copy_files, run_learning)
