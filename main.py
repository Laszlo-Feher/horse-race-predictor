import time

from feature_extractor import *
from machine_learning import *


def main():
    start = time.time()
    feature_vectors = extract_and_format_data(30)
    score = learn_and_test(feature_vectors, RES_TARGET)
    # score = 0
    print('\n')
    end = time.time()
    print("Time Usage: " + str(round((end - start), 2)) + " in seconds")
    print('\n')
    print("Test Accuracy: " + str(score))
    return 0


main()

