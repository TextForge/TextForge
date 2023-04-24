import numpy as np
from util.computeArrayStatistics import generateArrayStatistics
from scipy.stats import describe
from scipy.stats import entropy
import time

def get_words_per_doc_features(dataframe):

    start_time = time.time()
    
    words_per_doc = [len(str(doc).split()) for doc in dataframe["text"]]

    output_list = []


    output_list = output_list + generateArrayStatistics(feature_name_prefix="Words per Doc",
                                                        description= "words per document",
                                                        category = "Words per Doc statistics",
                                                        input_list =  words_per_doc)
    


    output_list.append({
        "feature": "Words per Doc Time",
        "value": time.time() - start_time,
        "description": "The time taken to calculate the words per doc features.",
        "category": "Words per Doc statistics"
    })


    return output_list