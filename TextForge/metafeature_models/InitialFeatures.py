import time

def get_initial_features(dataframe):
    start_time = time.time()
    output_list = []

    # Concatenate all text into a single string
    all_text = " ".join(dataframe["text"].map(str))
    
    # Split all_text into individual words and store in all_words list
    all_words = all_text.split()

    # Number of documents
    n_docs = len(dataframe)
    output_list.append({
        
        "feature": "Number of Documents",
        "value": n_docs,
        "description": "Number of documents in the dataset",
        "category": "Main 3 features"
    })

    # Number of categories
    n_cats = len(set(dataframe["label"]))
    output_list.append({
        
        "feature": "Number of Categories",
        "value": n_cats,
        "description": "Number of categories in the dataset",
        "category": "Main 3 features"
    })

    # Average word length

    #check if len(all_words) is 0 
    if len(all_words) > 0:
        avg_word_length = sum(len(word) for word in all_words) / len(all_words)
    else:
        avg_word_length = 0


    output_list.append({
        
        "feature": "Average Word Length",
        "value": avg_word_length,
        "description": "Average word length in the dataset",
        "category": "Main 3 features"
    })

    # Main 3 features time
    main_3_features_time = time.time() - start_time
    output_list.append({
        
        "feature": "Main 3 features Time",
        "value": main_3_features_time,
        "description": "The time taken to calculate the main 3 features.",
        "category": "Main 3 features"
    })

    return output_list
