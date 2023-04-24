import time

import nltk
import pyphen
from nltk.corpus import cmudict
from textblob import TextBlob

from util.computeArrayStatistics import generateArrayStatistics

# Load the English language tokenizer from nltk
tokenizer = nltk.tokenize.TreebankWordTokenizer()

# Load the English language hyphenation dictionary from pyphen
dic = pyphen.Pyphen(lang='en')

# Define a function to count the number of syllables in a word
def count_syllables(word):
    cmu_dict = cmudict.dict()

    # get the pronunciation of the word from the CMU Pronouncing Dictionary
    if word.lower() not in cmu_dict:
        return -1  # return -1 if the word is not found in the dictionary
    pronunciation = cmu_dict[word.lower()][0]

    # count the number of syllables in the pronunciation
    return len([s for s in pronunciation if s[-1].isdigit()])


# Define a function to calculate the average number of syllables per word in a text
def avg_syllables_per_word(text):
    
    return 0


def get_text_feature_taxonomy_metrics(dataframe):

    start_time = time.time()

    output_list = []


    all_text = " ".join(dataframe["text"].map(str))
    
    all_words = []

    # add all the words in all_text to all_words
    for word in all_text.split():
        all_words.append(word)



    #Frequency of character "@"
    output_list.append({
        "feature": "Frequency of character '@'",
        "value": all_text.count("@")/len(all_text),
        "description": "Frequency of character '@' in the dataset",
        "category": "Text Feature Taxonomy",
        "citation": "Zheng et al. (2006)"
    })

    #Average number of syllables per word
    output_list.append({
        
        "feature": "Average number of syllables per word",
        "value": avg_syllables_per_word(all_text),
        "description": "Average number of syllables per word in the dataset",
        "category": "Text Feature Taxonomy",
        "citation": "Feng et al. (2010)"
    })

    # Average sentence length in words
    #? Suh (2016)
    output_list.append({
        
        "feature": "Average sentence length in words",
        "value": sum(len(sentence.split()) for sentence in all_text.split("."))/len(all_text.split(".")),
        "description": "Average sentence length in words in the dataset",
        "category": "Text Feature Taxonomy",
        "citation": "Suh (2016)"
    })

    #Readability indices (Flesch, Kincaid, etc.)
    #? Flesch (1943); Kincaid (1975)


    # Vocabulary richness (e.g. Yule's K)
    #? Zheng et al. (2006); Suh (2016)
    # print("Vocabulary richness (e.g. Yule's K) is not implemented yet")

    # Fraction of past-tense verbs
    #? Jijkoun et al. (2010)
    output_list.append({
        
        "feature": "Fraction of past-tense verbs",
        "value": len([word for word in all_words if word.endswith("ed")])/len(all_words),
        "description": "Fraction of past-tense verbs in the dataset",
        "category": "Text Feature Taxonomy",
        "citation": "Jijkoun et al. (2010)"
    })


    # Mean number of noun phrases per sentence
    #? Chen and Zechner (2011)
    output_list.append({
        
        "feature": "Mean number of noun phrases per sentence",
        "value": sum(len(sentence.split()) for sentence in all_text.split("."))/len(all_text.split(".")),
        "description": "Mean number of noun phrases per sentence in the dataset",
        "category": "Text Feature Taxonomy",
        "citation": "Chen and Zechner (2011)"
    })



    # Mean number of parsing tree levels per sentence
    #? Chen and Zechner (2011) Massung et al. (2013)
    output_list.append({
        
        "feature": "Mean number of parsing tree levels per sentence",
        "value": sum(len(sentence.split()) for sentence in all_text.split("."))/len(all_text.split(".")),
        "description": "Mean number of parsing tree levels per sentence in the dataset",
        "category": "Text Feature Taxonomy",
        "citation": "Chen and Zechner (2011) Massung et al. (2013)"
    })

    # Contextual compatibility score
    #? Liao and Grishman (2010)
    output_list.append({
        
        "feature": "Contextual compatibility score",
        "value": sum(len(sentence.split()) for sentence in all_text.split("."))/len(all_text.split(".")),
        "description": "Contextual compatibility score in the dataset",
        "category": "Text Feature Taxonomy",
        "citation": "Liao and Grishman (2010)"
    })

    # Word polarity
    #? Turney (2002); Rice and Zorn (2013); Agarwal and Mittal (2016)
    output_list = output_list + generateArrayStatistics(feature_name_prefix="Word polarity",
                                                    description= "Word polarity in the dataset",
                                                    category = "Text Feature Taxonomy",
                                                    input_list = [TextBlob(word).sentiment.polarity for word in all_text.split()])
    


    print("Word polarity is not implemented yet")

    # Sentence polarity
    #? Missen et al. (2013)
    output_list = output_list + generateArrayStatistics(feature_name_prefix="Sentence polarity",
                                                    description= "Sentence polarity in the dataset",
                                                    category = "Text Feature Taxonomy",
                                                    input_list = [TextBlob(sentence).sentiment.polarity for sentence in all_text.split('.') if sentence])

    # Document Polarity(Review  Polarity)
    #? Mukherjee and Bhattacharyya (2012)
    output_list = output_list + generateArrayStatistics(feature_name_prefix="Document Polarity(Review  Polarity)",
                                                        description= "Document Polarity(Review  Polarity) in the dataset",
                                                        category = "Text Feature Taxonomy",
                                                        input_list = [TextBlob(document).sentiment.polarity for document in dataframe['text']])



    # Bag-of-words with only subjective/objective and positive/negative verbs
    #? Chesley et al. (2006)
    output_list.append({
        
        "feature": "Bag-of-words with only subjective/objective and positive/negative verbs",
        "value": sum(len(sentence.split()) for sentence in all_text.split("."))/len(all_text.split(".")),
        "description": "Bag-of-words with only subjective/objective and positive/negative verbs in the dataset",
        "category": "Text Feature Taxonomy"
    })

    output_list.append({
        
        "feature": "Text Feature Taxonomy Time",
        "value": time.time() - start_time,
        "description": "Time taken to calculate Text Feature Taxonomy features",
        "category": "Text Feature Taxonomy"
    })
     


    return output_list