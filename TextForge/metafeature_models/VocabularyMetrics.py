import numpy as np
import nltk
from util.computeArrayStatistics import generateArrayStatistics
from scipy.stats import describe
from scipy.stats import entropy
import time

def get_vocabulary_metrics(dataframe):
    start_time = time.time()
    all_text = " ".join(dataframe["text"].map(str))
    all_words = all_text.split()
    words_distribution = nltk.FreqDist(all_words)

    words_list_freq = [words_distribution[sample] for sample in words_distribution]
    words_list_freq.sort(reverse=True)
    terms_num = len(words_list_freq)
    terms_total = sum(words_list_freq)

    zipf_total = sum([1/i for i in range(1, terms_num+1)])
    term_probability = [words_list_freq[i]/terms_total for i in range(terms_num)]
    zipf_distribution = [(1/(i+1))/zipf_total for i in range(terms_num)]
    SEM = sum([term_probability[i] * np.log2(term_probability[i]/zipf_distribution[i]) for i in range(terms_num)])

    output_list = [{
        
        "feature": "Vocabulary SEM",
        "value": SEM,
        "description": "A measure of the difference between the distribution of the number of times each word appears in the text and a Zipf distribution.",
        "category": "Vocabulary statistics"
    }]
    

    output_list = output_list + generateArrayStatistics(feature_name_prefix="Vocabulary",
                                                        description= "Vocabulary per document",
                                                        category = "Vocabulary statistics",
                                                        input_list =  words_list_freq)
    




    output_list.append({
        
        "feature": "Vocabulary statistics Time",
        "value": time.time() - start_time,
        "description": "The time taken to calculate the vocabulary statistics features.",
        "category": "Vocabulary statistics"
    })


    return output_list
