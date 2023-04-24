import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)

# This feature is inspired by the AutoSklearn paper (Feurer et al., 2015)

def get_automl_metafeatures(dataframe):

    start_time = time.time()

    # Create CountVectorizer with max_features=5000
    count_vec = CountVectorizer(max_features=5000)

    # Create TfidfVectorizer with max_features=5000
    tfidf_vec = TfidfVectorizer(max_features=5000)

    # Create TfVectorizer with max_features=5000
    tf_vec = TfidfTransformer(use_idf=False, norm='l2', sublinear_tf=True)

    # Fit and transform the text column of the dataframe using each of the vectorizers
    count_features = count_vec.fit_transform(dataframe['text'])
    tfidf_features = tfidf_vec.fit_transform(dataframe['text'])
    tf_features = tf_vec.fit_transform(count_features)

    # Concatenate the features from all three vectorizers
    concatenated_features = hstack([count_features, tfidf_features, tf_features])

    # Convert the concatenated features to a pandas dataframe
    concatenated_features_df = pd.DataFrame(concatenated_features.toarray())

    # Add the label column to the concatenated features dataframe
    concatenated_features_df['label'] = dataframe['label']

    dataframe = concatenated_features_df

    num_patterns = dataframe.shape[0]
    num_classes = len(dataframe['label'].unique())
    num_features = dataframe.shape[1] - 1
    num_patterns_with_missing_values = dataframe.isnull().any(axis=1).sum()
    num_features_with_missing_values = dataframe.isnull().any(axis=0).sum()
    num_numeric_features = dataframe.select_dtypes(include=['float', 'int']).shape[1]
    num_categorical_features = dataframe.select_dtypes(include=['object']).apply(pd.Series.nunique)
    num_categorical_values = num_categorical_features.sum()
    ratio_numeric_to_categorical = num_numeric_features / num_categorical_features.sum()
    ratio_categorical_to_numeric = num_categorical_features.sum() / num_numeric_features
    
    # calculate log and total
    log_num_patterns = np.log(num_patterns)
    log_num_features = np.log(num_features)
    total_categorical_values = num_categorical_values
    
    # calculate percentage
    percent_patterns_with_missing_values = (num_patterns_with_missing_values / num_patterns) * 100
    percent_features_with_missing_values = (num_features_with_missing_values / num_features) * 100

    # calculate kurtosis and skewness for missing values
    kurtosis_missing_values = dataframe.isnull().kurtosis().min()
    skewness_missing_values = dataframe.isnull().skew().max()
    
    # calculate kurtosis and skewness for numeric features
    kurtosis_numeric_features = dataframe.select_dtypes(include=['float', 'int']).kurtosis()
    skewness_numeric_features = dataframe.select_dtypes(include=['float', 'int']).skew()
    
    # calculate pca metafeatures
    pca = PCA(n_components=0.95)
    pca.fit(dataframe.drop('label', axis=1))
    pca_dimensionality = pca.n_components_
    pca_skewness = skewness_numeric_features[pca_dimensionality - 1]
    pca_kurtosis = kurtosis_numeric_features[pca_dimensionality - 1]
    
    # calculate inverse pca dimensionality
    inv_pca_dimensionality = 1 / pca_dimensionality
    log_inv_pca_dimensionality = np.log(inv_pca_dimensionality)
    
    output_list = []

    output_list.append({
        
        "feature": "Number of Patterns",
        "value": num_patterns,
        "description": "The number of patterns in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Log Number of Patterns",
        "value": log_num_patterns,
        "description": "The logarithm of the number of patterns in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Number of Classes",
        "value": num_classes,
        "description": "The number of classes in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Number of Features",
        "value": num_features,
        "description": "The number of features in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Log Number of Features",
        "value": log_num_features,
        "description": "The logarithm of the number of features in the dataset.",
        "category": "AutoSklearn features"
    })


    output_list.append({
        
        "feature": "Kurtosis of Missing Values",
        "value": kurtosis_missing_values,
        "description": "The kurtosis of the missing values in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Skewness of Missing Values",
        "value": skewness_missing_values,
        "description": "The skewness of the missing values in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Minimum Kurtosis of Numeric Features",
        "value": kurtosis_numeric_features.min(),
        "description": "The minimum kurtosis of the numeric features in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Maximum Kurtosis of Numeric Features",
        "value": kurtosis_numeric_features.max(),
        "description": "The maximum kurtosis of the numeric features in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Mean Kurtosis of Numeric Features",
        "value": kurtosis_numeric_features.mean(),
        "description": "The mean kurtosis of the numeric features in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Standard Deviation of Kurtosis of Numeric Features",
        "value": kurtosis_numeric_features.std(),
        "description": "The standard deviation of the kurtosis of the numeric features in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Minimum Skewness of Numeric Features",
        "value": skewness_numeric_features.min(),
        "description": "The minimum skewness of the numeric features in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Maximum Skewness of Numeric Features",
        "value": skewness_numeric_features.max(),
        "description": "The maximum skewness of the numeric features in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Mean Skewness of Numeric Features",
        "value": skewness_numeric_features.mean(),
        "description": "The mean skewness of the numeric features in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Standard Deviation of Skewness of Numeric Features",
        "value": skewness_numeric_features.std(),
        "description": "The standard deviation of the skewness of the numeric features in the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "PCA Dimensionality",
        "value": pca_dimensionality,
        "description": "The dimensionality of the PCA of the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "PCA Skewness",
        "value": pca_skewness,
        "description": "The skewness of the PCA of the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "PCA Kurtosis",
        "value": pca_kurtosis,
        "description": "The kurtosis of the PCA of the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Inverse PCA Dimensionality",
        "value": inv_pca_dimensionality,
        "description": "The dimensionality of the inverse PCA of the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "Log Inverse PCA Dimensionality",
        "value": log_inv_pca_dimensionality,
        "description": "The log of the dimensionality of the inverse PCA of the dataset.",
        "category": "AutoSklearn features"
    })

    output_list.append({
        
        "feature": "AutoSklearn features Time",
        "value": time.time() - start_time,
        "description": "The time it took to compute the AutoSklearn features.",
        "category": "AutoSklearn features"
    })

    return output_list