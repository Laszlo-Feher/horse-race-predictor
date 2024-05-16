from machine_learning.classification import classification_with_bulk_fvs, classification_with_individual_results, \
    classification_with_equal_results, split_to_first_3_and_the_rest, classify_by_race_without_conversion
from machine_learning.pairwise_ranking import pairwise_learn_to_rank_pairwise, pairwise_learn_to_rank_ndcg, pairwise_learn_to_rank_map

def learn_and_test(df, target, algorythm, formatted_time):
    result = 'no results'

    match algorythm:
        case 'all':
            df_bulk_fvs_copy = df.copy()
            classification_with_bulk_fvs(df_bulk_fvs_copy, target, formatted_time)

            df_individual_results_copy = df.copy()
            classification_with_individual_results(df_individual_results_copy, target, formatted_time)

            df_equal_results_copy = df.copy()
            classification_with_equal_results(df_equal_results_copy, target, formatted_time)

            df_split_copy = df.copy()
            split_to_first_3_and_the_rest(df_split_copy, target, formatted_time)

            df_race_copy = df.copy()
            classify_by_race_without_conversion(df_race_copy, target, formatted_time)

            df_pairwise_copy = df.copy()
            pairwise_learn_to_rank_pairwise(df_pairwise_copy, target, formatted_time)

            df_ndcg_copy = df.copy()
            return pairwise_learn_to_rank_ndcg(df_ndcg_copy, target, formatted_time)
            # return pairwise_learn_to_rank_map(df, target, formatted_time)
        case "classification_with_bulk_fvs":
            return classification_with_bulk_fvs(df, target, formatted_time)
        case "classification_with_individual_results":
            return classification_with_individual_results(df, target, formatted_time)
        case "classification_with_equal_results":
            return classification_with_equal_results(df, target, formatted_time)
        case "split_to_first_3_and_the_rest":
            return split_to_first_3_and_the_rest(df, target, formatted_time)
        case "classify_by_race_without_conversion":
            return classify_by_race_without_conversion(df, target, formatted_time)

        case "pairwise_learn_to_rank_pairwise":
            return pairwise_learn_to_rank_pairwise(df, target, formatted_time)
        case "pairwise_learn_to_rank_ndcg":
            return pairwise_learn_to_rank_ndcg(df, target, formatted_time)
        case _:
            print("Not an implemented method!")

    return result
