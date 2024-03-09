import time

from feature_extractor import *
from machine_learning import *

# TODO add more, and set automatic
# kiszervezni ml algrotimusok es teljes parameterhetoseg
# mainbe lefuttatni osszes opcio varianst es eremenyeket osszehasonlitani
# random_forest_classifier: {
#     'is_divided_to_races': False,
#     'convert_to_binary': True,
# }

# ellenőrizni training data nelkul és osszehasonlitani az eredmenyt
# kiiratni mennyi fajl ment at
# késobb raceket is ellenorizni lehet


def main():
    start = time.time()
    feature_vectors = extract_and_format_data(150, False, True)
    score = learn_and_test(feature_vectors, RES_TARGET, "", True)
    # score = 0
    print('\n')
    end = time.time()
    print("Time Usage: " + str(round((end - start), 2)) + " in seconds")
    print('\n')
    print("Test Accuracy: " + str(score))
    return 0


main()
