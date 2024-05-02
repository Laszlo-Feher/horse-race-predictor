from machine_learning.classification import classification_with_bulk_fvs, classification_with_individual_results, \
    classification_with_equal_results, split_to_first_3_and_the_rest
from machine_learning.pairwise_ranking import pairwise_learn_to_rank


def listwise_learn_to_rank(df, target):
    return None


def pointwise_learn_to_rank(df, target):
    return None


def learn_and_test(df, target, algorythm):
    result = 'no results'

    match algorythm:
        case "classification_with_bulk_fvs":
            return classification_with_bulk_fvs(df, target)
        case "classification_with_individual_results":
            return classification_with_individual_results(df, target)
        case "classification_with_equal_results":
            return classification_with_equal_results(df, target)
        case "split_to_first_3_and_the_rest":
            return split_to_first_3_and_the_rest(df, target)

        case "pairwise_learn_to_rank":
            return pairwise_learn_to_rank(df, target)
        case "listwise_learn_to_rank":
            return listwise_learn_to_rank(df, target)
        case "pointwise_learn_to_rank":
            return pointwise_learn_to_rank(df, target)
        case _:
            print("Not an implemented method!")

    return result
