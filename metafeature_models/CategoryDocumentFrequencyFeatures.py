
from util.computeArrayStatistics import generateArrayStatistics
from scipy.stats import entropy
from scipy.stats import describe
import numpy as np

import time

def get_category_document_frequency_features(dataframe):

    start_time = time.time()

    output_list = []

    docs_per_category = []

    for category in set(dataframe["label"]):
        docs_num = dataframe[dataframe["label"] == category].shape[0]
        docs_per_category.append(docs_num)


    output_list = output_list + generateArrayStatistics(feature_name_prefix="Documents Per Category",
                                                        description= "docuemnts per category",
                                                        category = "Docs per Category statistics",
                                                        input_list =  docs_per_category)
    
    output_list.append({
        "feature": "DPC Time",
        "value": time.time() - start_time,
        "description": "The time taken to calculate the docs per category features.",
        "category": "Docs per Category statistics"
    })


    return output_list