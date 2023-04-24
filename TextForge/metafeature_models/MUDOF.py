import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from TextForge.util.computeArrayStatistics import generateArrayStatistics

import time

def get_MUDOF_features(dataframe):

    start_time = time.time()

    train_data, tune_data = train_test_split(dataframe, test_size=0.3, random_state=42)

    #? The First 2 metafeatures are not relavant to us, so we will skip them
    # split the dataframe into positive training and tuning examples for each label
    pos_tr_examples = {}
    pos_tu_examples = {}
    for label in dataframe["label"].unique():
        pos_tr_examples[label] = train_data[(train_data["label"] == label)]
        pos_tu_examples[label] = tune_data[(tune_data["label"] == label)]

    # calculate PosTr and PosTu
    PosTr = {label: len(pos_tr_examples[label]) for label in dataframe["label"].unique()}
    PosTu = {label: len(pos_tu_examples[label]) for label in dataframe["label"].unique()}


    # calculate AvgDocLen
    avg_doc_len = {}
    for label in dataframe["label"].unique():
        docs = pos_tr_examples[label]["text"]
        vectorizer = TfidfVectorizer(use_idf=False, norm=None)
        X = vectorizer.fit_transform(docs)
        doc_lens = X.sum(axis=1)
        avg_doc_len[label] = float(doc_lens.mean())

    # calculate AvgTermVal, AvgMaxTermVal, AvgMinTermVal, and AvgTermThre
    avg_term_val = {}
    avg_max_term_val = {}
    avg_min_term_val = {}
    avg_term_thre = {}

    for label in dataframe["label"].unique():
        docs = pos_tr_examples[label]["text"]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(docs)

        # calculate AvgTermVal
        doc_term_vals = X.mean(axis=1)
        avg_term_val[label] = float(doc_term_vals.mean())

        # calculate AvgMaxTermVal
        doc_max_term_vals = X.max(axis=1)
        avg_max_term_val[label] = float(doc_max_term_vals.mean())

        # calculate AvgMinTermVal
        doc_min_term_vals = X.min(axis=1)
        avg_min_term_val[label] = float(doc_min_term_vals.mean())

        # calculate AvgTermThre
        term_thre = 0.5  # set the term weight threshold
        n_terms_above_thre = (X > term_thre).sum(axis=1)
        avg_term_thre[label] = float(n_terms_above_thre.mean())

    # calculate AvgTopInfoGain and NumInfoGainThres
    avg_top_info_gain = {}
    num_info_gain_thres = {}

    for label in dataframe["label"].unique():
        docs = pos_tr_examples[label]["text"]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(docs)

        # calculate information gain for each term
        info_gain = X.multiply(X.log1p()).sum(axis=0)
        info_gain = np.array(info_gain).squeeze()
        sorted_indices = np.argsort(-info_gain)

        # calculate AvgTopInfoGain
        top_m = 10  # set the number of top terms to consider
        top_info_gain = info_gain[sorted_indices[:top_m]].mean()
        avg_top_info_gain[label] = float(top_info_gain)

        # calculate NumInfoGainThres
        info_gain_thre = 0.1  # set the information gain threshold
        n_terms_above_thre = (info_gain > info_gain_thre).sum()
        num_info_gain_thres[label] = n_terms_above_thre



    output_list = []



    output_list = output_list + generateArrayStatistics(feature_name_prefix="PosTr",
                                                        description= "positive training examples of a category.",
                                                        category = "MUDOF",
                                                        input_list =  list(PosTr.values()))
    

    output_list = output_list + generateArrayStatistics(feature_name_prefix="PosTu",
                                                        description= "positive tuning examples of a category.",
                                                        category = "MUDOF",
                                                        input_list =  list(PosTu.values()))
    
    
    output_list = output_list + generateArrayStatistics(feature_name_prefix="AvgDocLen",
                                                        description= "The average document length of a category.",
                                                        category = "MUDOF",
                                                        input_list =  list(avg_doc_len.values()))
    
    output_list = output_list + generateArrayStatistics(feature_name_prefix="AvgTermVal",
                                                        description= "The average term value of a category.",
                                                        category = "MUDOF",
                                                        input_list =  list(avg_term_val.values()))
    
    output_list = output_list + generateArrayStatistics(feature_name_prefix="AvgMaxTermVal",
                                                        description= "The average maximum term value of a category.",
                                                        category = "MUDOF",
                                                        input_list =  list(avg_max_term_val.values()))
    
    output_list = output_list + generateArrayStatistics(feature_name_prefix="AvgMinTermVal",
                                                        description= "The average minimum term value of a category.",
                                                        category = "MUDOF",
                                                        input_list =  list(avg_min_term_val.values()))
    
    output_list = output_list + generateArrayStatistics(feature_name_prefix="AvgTermThre",
                                                        description= "The average number of terms above a threshold of a category.",
                                                        category = "MUDOF",
                                                        input_list =  list(avg_term_thre.values()))
    
    output_list = output_list + generateArrayStatistics(feature_name_prefix="AvgTopInfoGain",
                                                        description= "The average information gain of the top m terms of a category.",
                                                        category = "MUDOF",
                                                        input_list =  list(avg_top_info_gain.values()))
    
    output_list = output_list + generateArrayStatistics(feature_name_prefix="NumInfoGainThres",
                                                        description= "The number of terms with information gain above a threshold of a category.",
                                                        category = "MUDOF",
                                                        input_list =  list(num_info_gain_thres.values()))
    

    output_list.append({
        
        "feature": "MUDOF Time",
        "value": time.time() - start_time,
        "description": "The time taken to calculate the MUDOF features.",
        "category": "MUDOF"
    })


    return output_list
