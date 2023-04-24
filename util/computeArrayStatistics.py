from scipy.stats import describe
from scipy.stats import entropy
import numpy as np






def calculateArrayStatistics(input_list):
    nobs, (min_input, max_input), mean_input, std_input, skew_input, kurtosis_input = describe(input_list)

    percentiles = [np.percentile(input_list, 25), np.percentile(input_list, 75)]

    std_input2 = np.sqrt(std_input)
    ratio_avg_sd_input_list = mean_input/std_input
    ratio_avg_sd_input_list2 = mean_input/std_input2

    entropy_input_list = entropy(input_list, base=2)

    log_nobs = np.log(nobs)

    # Combine the results into a single list and return it
    output_list = [
        {"suffix": "nobs", "value": nobs},
        {"suffix": "log_nobs", "value": log_nobs},
        {"suffix": "min", "value": min_input},
        {"suffix": "max", "value": max_input},
        {"suffix": "mean", "value": mean_input},
        {"suffix": "std", "value": std_input},
        {"suffix": "std_sqrt", "value": std_input2},
        {"suffix": "skew", "value": skew_input},
        {"suffix": "kurtosis", "value": kurtosis_input},
        {"suffix": "avg_div_sd", "value": ratio_avg_sd_input_list},
        {"suffix": "avg_div_sd_sqrt", "value": ratio_avg_sd_input_list2},
        {"suffix": "entropy", "value": entropy_input_list},
        {"suffix": "percentile_25", "value": percentiles[0]},
        {"suffix": "percentile_75", "value": percentiles[1]}
    ]

    return output_list


def generateArrayStatistics(feature_name_prefix, description, category, input_list):
    output_list = []

    for value in calculateArrayStatistics(input_list):
        suffix = value["suffix"]
        feature_name = f"{feature_name_prefix} - {suffix}"
        feature_value = value["value"]
        feature_description = f"The {suffix} of the {description}"
        feature_category = category

        output_list.append({
            "feature": feature_name,
            "value": feature_value,
            "description": feature_description,
            "category": feature_category
        })

    return output_list
