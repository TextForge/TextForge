import pandas as pd

import string

def convert_classificaion_dataset_to_features(dataframe):
    df = dataframe
    # print("../3Backend/project/Datasets - Train/"+file_name)
    # print(df)
    #remove rows with any column having null values
    df = df.dropna()

    features_obj = {}

    print("size")

    features_obj['size'] = df.shape[0]
    features_obj['columns'] = df.shape[1]
    features_obj['unique_labels'] = df['label'].nunique()
    features_obj['null_values'] = df.isnull().sum().sum()

    print("imbalance")

    features_obj['imbalance'] = df['label'].value_counts().max() / df['label'].value_counts().min()

    features_obj['mean_text_length'] = df['text'].str.len().mean()
    features_obj['max_text_length'] = df['text'].str.len().max()
    features_obj['min_text_length'] = df['text'].str.len().min()
    features_obj['std_text_length'] = df['text'].str.len().std()
    features_obj['median_text_length'] = df['text'].str.len().median()
    features_obj['skew_text_length'] = df['text'].str.len().skew()
    # features_obj['kurtosis_text_length'] = df['text'].str.len().kurtosis()
    features_obj['variance_text_length'] = df['text'].str.len().var()

    #? Only for 2 Classes (eg: Correlation between length and label)
    # features_obj['covariance_text_length'] = df['text'].str.len().cov()
    # features_obj['correlation_text_length'] = df['text'].str.len().corr()
    # features_obj['quantile_text_length'] = df['text'].str.len().quantile()

    print("mean,max,min,skew")

    features_obj['mean_text_word_count'] = df['text'].str.split().str.len().mean()
    features_obj['max_text_word_count'] = df['text'].str.split().str.len().max()
    features_obj['min_text_word_count'] = df['text'].str.split().str.len().min()
    features_obj['std_text_word_count'] = df['text'].str.split().str.len().std()
    features_obj['median_text_word_count'] = df['text'].str.split().str.len().median()
    features_obj['skew_text_word_count'] = df['text'].str.split().str.len().skew()
    # features_obj['kurtosis_text_word_count'] = df['text'].str.split().str.len().kurtosis()
    features_obj['variance_text_word_count'] = df['text'].str.split().str.len().var()

    #? Only for 2 Classes (eg: Correlation between length and label)
    # features_obj['covariance_text_word_count'] = df['text'].str.split().str.len().cov()
    # features_obj['correlation_text_word_count'] = df['text'].str.split().str.len().corr()
    # features_obj['quantile_text_word_count'] = df['text'].str.split().str.len().quantile()
   
    # print(df['text'])
    #convert df['text'] to 1 long string
    
    # txt = df['text'].str.cat(sep=' ')
    # txt = df['text']
    # vect.fit_transform(df['text'])
    # test_X_dtm = vect.transform(df['text'])

    #create new dictionary with the number of time each word appears in the dataset text column

    print("unique_words")

    # percentage of unique words in the text
    features_obj['unique_words'] = df['text'].apply(lambda x: len(set(str(x).split())))

    print("text")
    #text
    features_obj['text'] = df['text'].str.cat(sep=' ')

    print("text_maj")
    #majority class text
    features_obj['majority_class_text'] = df[df['label'] == df['label'].value_counts().idxmax()]['text'].str.cat(sep=' ')

    print("text_min")
    #minority class text
    features_obj['minority_class_text'] = df[df['label'] == df['label'].value_counts().idxmin()]['text'].str.cat(sep=' ')

    

    print("capital_words")
    #percentage of capital words in the text

    features_obj['capital_words'] = df['text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))


    #percentage of stop words in the text
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    features_obj['stop_words'] = df['text'].apply(lambda x: len([x for x in x.split() if x in stop]))

    #percentage of punctuations in the text
    features_obj['punctuations'] = df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    #percentage of numerics in the text
    features_obj['numerics'] = df['text'].apply(lambda x: len([c for c in str(x) if c in string.digits]))

    #percentage of upper case words in the text
    features_obj['upper'] = df['text'].apply(lambda x: len([c for c in str(x).split() if c.isupper()]))

    #percentage of title case words in the text
    features_obj['title'] = df['text'].apply(lambda x: len([c for c in str(x).split() if c.istitle()]))

    print("percentiles")

    #percentage of words with more than 3-15 characters in the text
    features_obj['words_more_than_3'] = df['text'].apply(lambda x: len([c for c in str(x).split() if len(c) > 3]))
    features_obj['words_more_than_5'] = df['text'].apply(lambda x: len([c for c in str(x).split() if len(c) > 5]))
    features_obj['words_more_than_7'] = df['text'].apply(lambda x: len([c for c in str(x).split() if len(c) > 7]))
    features_obj['words_more_than_9'] = df['text'].apply(lambda x: len([c for c in str(x).split() if len(c) > 9]))
    features_obj['words_more_than_11'] = df['text'].apply(lambda x: len([c for c in str(x).split() if len(c) > 11]))
    features_obj['words_more_than_13'] = df['text'].apply(lambda x: len([c for c in str(x).split() if len(c) > 13]))
    features_obj['words_more_than_15'] = df['text'].apply(lambda x: len([c for c in str(x).split() if len(c) > 15]))


    # print(features_obj)

    #convert the features object to a dataframe
    features_df = pd.DataFrame(features_obj, index=[0])
    return features_df
