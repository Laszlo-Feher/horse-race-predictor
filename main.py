import time

from feature_extractor import *
from file_filterer import filter_files_and_copy
from file_writer import *
from machine_learning import *


def main(copy_files, run_learning):
    if copy_files:
        filter_files_and_copy()

    if run_learning:
        start = time.time()
        feature_vectors = extract_and_format_data(6589, False, True)
        if feature_vectors is None:
            print("Something went wrong during data extracting and formatting!")
            return 0

        score = learn_and_test(feature_vectors, RES_TARGET, "classification_with_equal_results", False)
        print('\n')
        end = time.time()
        print("Time Usage: " + str(round((end - start), 2)) + " in seconds")
        print('\n')
        print("Test Accuracy: " + str(score))

    return 0


copy_files = False
run_learning = True

main(copy_files, run_learning)
