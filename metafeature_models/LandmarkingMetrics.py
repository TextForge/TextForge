from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from sklearn.metrics import (accuracy_score, balanced_accuracy_score, top_k_accuracy_score, 
                             average_precision_score, brier_score_loss, f1_score, 
                             precision_score, recall_score, jaccard_score, roc_auc_score)

import time

def calculate_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
  
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')

    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')

    jaccard_micro = jaccard_score(y_true, y_pred, average='micro')
    jaccard_macro = jaccard_score(y_true, y_pred, average='macro')
    jaccard_weighted = jaccard_score(y_true, y_pred, average='weighted')
    
    lst = [
        {"suffix": "accuracy", "value": accuracy},
        {"suffix": "f1_micro", "value": f1_micro},
        {"suffix": "f1_macro", "value": f1_macro},
        {"suffix": "f1_weighted", "value": f1_weighted},
        {"suffix": "precision_micro", "value": precision_micro},
        {"suffix": "precision_macro", "value": precision_macro},
        {"suffix": "precision_weighted", "value": precision_weighted},
        {"suffix": "recall_micro", "value": recall_micro},
        {"suffix": "recall_macro", "value": recall_macro},
        {"suffix": "recall_weighted", "value": recall_weighted},
        {"suffix": "jaccard_micro", "value": jaccard_micro},
        {"suffix": "jaccard_macro", "value": jaccard_macro},
        {"suffix": "jaccard_weighted", "value": jaccard_weighted}
    ]

    output_list = []

    for metric in lst:
        output_list.append({
            "feature": model_name+ " Landmarking " + metric['suffix'],
            "value": metric['value'],
            "description": "The " + metric['suffix'] + " of the " + model_name + " model",
            "category": "Landmarking"
        })

    return output_list

def get_landmarking_metrics(dataframe):

    start_time = time.time()

    output_list = []

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(dataframe['text'])
    y = dataframe['label']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

    knn_clsf = KNeighborsClassifier(n_neighbors=1)
    knn_clsf.fit(X_train, y_train)
    predictions = knn_clsf.predict(X_test)
    onenn = calculate_metrics(y_test, predictions, "KNeighborsClassifier")
    predictions = []
    
    tree_clsf = DecisionTreeClassifier()
    tree_clsf.fit(X_train, y_train)
    predictions = tree_clsf.predict(X_test)
    tree = calculate_metrics(y_test, predictions, "DecisionTreeClassifier")
    predictions = []
    
    lin_clsf = LogisticRegression(solver='saga', n_jobs=2)
    lin_clsf.fit(X_train, y_train)
    predictions = lin_clsf.predict(X_test)
    lin = calculate_metrics(y_test, predictions, "LogisticRegression")
    predictions = []
    
    nb_clsf = MultinomialNB()
    nb_clsf.fit(X_train, y_train)
    predictions = nb_clsf.predict(X_test)
    nb = calculate_metrics(y_test, predictions, "MultinomialNB")

    output_list = output_list + onenn + tree + lin + nb

    output_list.append({
        "feature": "Landmarking Time",
        "value": time.time() - start_time,
        "description": "Time to calculate landmarking features",
        "category": "Landmarking"
    })

    return output_list