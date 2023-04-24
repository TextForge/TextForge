
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import describe
from util.computeArrayStatistics import generateArrayStatistics

import time

def get_principal_component_analysis_metafeatures(dataframe):

    start_time = time.time()

    output_list = []

    #BoW Statistics
    pcac_bow = 0
    pca_singular_sum_bow = 0
    pca_explained_ratio_bow = 0
    pca_explained_var_bow = 0
    pcap_bow = 0
    zero_pct = 0
    
    document_vocabulary_size = []

    for doc in dataframe["text"].map(str):
        doc_words = doc.split()
        doc_words = set(doc_words) # number of unique words in this document
        document_vocabulary_size.append(len(doc_words))



    category_vocabulary_size = []
    category_vocabulary = []

    #for unique category
    for category in dataframe["label"].unique():
        category_vocabulary.append([])
        for doc in dataframe[dataframe["label"] == category]["text"]:
            doc_words = str(doc).split()
            category_vocabulary[-1].extend(doc_words)
        category_vocabulary[-1] = set(category_vocabulary[-1])
        category_vocabulary_size.append(len(category_vocabulary[-1]))

    # print("vocabulary_size")
    # print(category_vocabulary_size)
    # print(category_vocabulary)


    #BoW statistics / traditional meta-features
    n_docs = len(document_vocabulary_size)
    count_vect = CountVectorizer(decode_error = 'ignore', max_features = 25000)
    

    X_train_counts = count_vect.fit_transform(dataframe['text'])
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    svd = TruncatedSVD(n_components=100)
    svd.fit(X_train_tf)
    pcap_bow = svd.explained_variance_[0]

    output_list.append({
        
        "feature": "PCAP",
        "value": pcap_bow,
        "description": "The first principal component of the BoW matrix.",
        "category": "Principal Components (PC) statistics"
    })


    output_list = output_list + generateArrayStatistics(feature_name_prefix="PCA",
                                                        description= "first principal component of the BoW matrix.",
                                                        category = "Principal Components (PC) statistics",
                                                        input_list =  svd.components_[0])
    





    for p in range(len(svd.explained_variance_)):
        if p < 10:
            pcac_bow += svd.singular_values_[p]
        pca_singular_sum_bow += svd.singular_values_[p]
        pca_explained_ratio_bow += svd.explained_variance_ratio_[p]
        pca_explained_var_bow += svd.explained_variance_[p]
    pcac_bow = pcac_bow / pca_singular_sum_bow


    output_list.append({
        
        "feature": "PCAC",
        "value": pcac_bow,
        "description": "The cumulative explained variance ratio of the first 10 principal components of the BoW matrix.",
        "category": "Principal Components (PC) statistics"
    })

    #pca_singular_sum_bow
    output_list.append({
        
        "feature": "PCASingularSum",
        "value": pca_singular_sum_bow,
        "description": "The sum of the singular values of the BoW matrix.",
        "category": "Principal Components (PC) statistics"
    })

    #pca_explained_ratio_bow
    output_list.append({
        
        "feature": "PCAExplainedRatio",
        "value": pca_explained_ratio_bow,
        "description": "The sum of the explained variance ratio of the BoW matrix.",
        "category": "Principal Components (PC) statistics"
    })

    #pca_explained_var_bow
    output_list.append({
        
        "feature": "PCAExplainedVar",
        "value": pca_explained_var_bow,
        "description": "The sum of the explained variance of the BoW matrix.",
        "category": "Principal Components (PC) statistics"
    })
    

    zero_pct = X_train_tf.getnnz()
    zero_pct = 1 - (zero_pct / (30000*n_docs))

    # print(zero_pct)

    output_list.append({
        
        "feature": "Zero Percentage",
        "value": zero_pct,
        "description": "The percentage of zero values in the document-term matrix.",
        "category": "Principal Components (PC) statistics"
    })

    # print("Time taken to compute principal component metafeatures: %s seconds" % (time.time() - start_time))
    output_list.append({
        
        "feature": "PC Time Taken",
        "value": time.time() - start_time,
        "description": "The time taken to compute the principal component metafeatures.",
        "category": "Principal Components (PC) statistics"
    })



    return output_list