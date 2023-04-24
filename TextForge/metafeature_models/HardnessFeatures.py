
import math
import nltk
import numpy as np
from scipy.stats import entropy


from scipy.stats import describe

import time

def get_hardness_features(dataframe):

    start_time = time.time()

    output_list = []

    all_text = " ".join(dataframe["text"].map(str))

    all_words = []

    # add all the words in all_text to all_words
    for word in all_text.split():
        all_words.append(word)

    n_docs = len(dataframe)

    document_vocabulary_size = []

    for doc in dataframe["text"]:
        doc_words = doc.split()
        doc_words = set(doc_words) # number of unique words in this document
        document_vocabulary_size.append(len(doc_words))


    #Hardness
    imbalance = 0 #Class imbalance
    SEM = 0 #Stylometric Evaluation Measure
    SVB = 0 #Supervised Vocabulary Based
    UVB = 0 #Unsupervised Vocabulary Based
    MRH_J = 0 #Macro-average Relative Hardness-Jaccard
    VL = 0 #Vocabulary Length
    VDR = 0 #Vocabulary Document Ratio

    # print('Harness meta-features...')
    
    #SEM
    words_list_freq = []
    words_distribution = nltk.FreqDist(all_words)
    #print(words_distribution.most_common(10))
    for sample in words_distribution:
        words_list_freq.append(words_distribution[sample])
    words_list_freq.sort(reverse=True)
    terms_num = len(words_list_freq) #Vocabulary size
    terms_total = sum(words_list_freq)
    zipf_total = 0
    for i in range(1,terms_num+1):
        zipf_total+=(1/i)
    term_probability = []
    zipf_distribution = []
    
    for i in range(terms_num):
        term_probability.append(words_list_freq[i]/terms_total)
        zipf_distribution.append((1/(i+1))/zipf_total)
        SEM += term_probability[i] * np.log2(term_probability[i]/zipf_distribution[i])
    
    #UVB
    #print("Number of docs???: "+str(n_docs))
    for doc in document_vocabulary_size:
        UVB += pow((doc - terms_num)/n_docs,2)
    UVB = UVB/n_docs
    UVB = math.sqrt(UVB)

    #get number of unique values in the label column
    number_of_categories = len(set(dataframe['label']))


    category_vocabulary_size = []
    category_vocabulary = []

    #for unique category
    for category in dataframe["label"].unique():
        category_vocabulary.append([])
        for doc in dataframe[dataframe["label"] == category]["text"]:
            doc_words = doc.split()
            category_vocabulary[-1].extend(doc_words)
        category_vocabulary[-1] = set(category_vocabulary[-1])
        category_vocabulary_size.append(len(category_vocabulary[-1]))

    # print("vocabulary_size")
    # print(category_vocabulary_size)
    # print(category_vocabulary)


    #SVB
    for category in category_vocabulary_size:
        SVB += pow((category - terms_num)/n_docs,2)
    SVB = SVB/number_of_categories
    SVB = math.sqrt(SVB)


    words_per_doc = []

    for doc in dataframe["text"]:
        doc_words = doc.split()
        words_num = len(doc_words) # number of words in this document
        words_per_doc.append(words_num)


#		mrhs = []
#		mrhi = []
#		mrhj = []
    #MRH_J
    for cati in range(number_of_categories-1):
        for catj in range(cati+1, number_of_categories):
            valor = len(category_vocabulary[cati].intersection(category_vocabulary[catj])) / len(category_vocabulary[cati] | category_vocabulary[catj])
#				mrhs.append(valor)
#				mrhi.append(cati)
#				mrhj.append(catj)
            MRH_J += valor
#		print('more related')
#		top5 = sorted(range(len(mrhs)), key=lambda i: mrhs[i])[-5:]
#		for i in top5:
#			print(str(mrhi[i]) + ' ' + str(mrhj[i]) + ' with: ' + str(mrhs[i]))

    #VL
    VL = terms_num / len(words_per_doc)



    words_per_doc = []

    for doc in dataframe["text"]:
        doc_words = doc.split()
        words_num = len(doc_words) # number of words in this document
        words_per_doc.append(words_num)

    nobs, minmax_wpd, avg_wpd, sd_wpd, skw_wpd, kur_wpd = describe(words_per_doc)
    
    #VDR
    VDR = np.log2(VL) / np.log2(avg_wpd)
    
    #Vocabulary stats
    size_voc, minmax_voc, avg_voc, sd_voc, skw_voc, kur_voc = describe(words_list_freq)
    sd_voc = np.sqrt(sd_voc)
    ratio_avg_sd_voc = avg_voc/sd_voc
    entropy_voc = entropy(words_list_freq, base = 2)
    




    output_list.append({
        
        "feature": "SEM",
        "value": SEM,
        "description": "Stylometric Evaluation Measure",
        "category": "Hardness"
    })

    output_list.append({
        
        "feature": "UVB",
        "value": UVB,
        "description": "Unsupervised Vocabulary Based",
        "category": "Hardness"
    })

    output_list.append({
        
        "feature": "SVB",
        "value": SVB,
        "description": "Supervised Vocabulary Based",
        "category": "Hardness"
    })

    output_list.append({
        
        "feature": "MRH_J",
        "value": MRH_J,
        "description": "Macro-average Relative Hardness-Jaccard",
        "category": "Hardness"
    })

    output_list.append({
        
        "feature": "VL",
        "value": VL,
        "description": "Vocabulary Length",
        "category": "Hardness"
    })

    output_list.append({
        
        "feature": "VDR",
        "value": VDR,
        "description": "Vocabulary Document Ratio",
        "category": "Hardness"
    })



    
    
    endc = len(dataframe)/number_of_categories

    #iterate over the unique labels
    for label in set(dataframe['label']):
        docs_number = len(dataframe[dataframe['label'] == label])
        imbalance += pow(docs_number - endc, 2)

    imbalance = np.sqrt(imbalance/number_of_categories)


    output_list.append({
        
        "feature": "Imbalance",
        "value": imbalance,
        "description": "Class imbalance",
        "category": "Hardness"
    })

    output_list.append({
        
        "feature": "Hardness Time",
        "value": time.time() - start_time,
        "description": "Time to calculate hardness features",
        "category": "Hardness"
    })
    



    return output_list
