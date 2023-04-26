import time
import textstat

def check_value(value, name):
    if not isinstance(value, (int, float)):
        return  # Ignore non-numeric values
    if value < -10 or value > 1000:
        raise ValueError(f"{value} is out of range (-10 to 1,000).")

def cap_the_values(value, min_possible_value, max_possible_value):
    if value < min_possible_value:
        return min_possible_value
    if value > max_possible_value:
        return max_possible_value
    return value

def get_textstat_metrics(dataframe):
    """

    https://pypi.org/project/textstat/
    https://github.com/textstat/textstat/


    Computes various text statistics for a given dataframe and returns them as a list of dictionaries.
    Each dictionary represents a different statistic and includes a name, value, and description.

    Args:
        dataframe: pandas DataFrame containing a 'text' column with text data.

    Returns:
        A list of dictionaries representing different text statistics.
    """
    
    start_time = time.time()

    # Concatenate all text into a single string
    all_text = " ".join(dataframe["text"].astype(str))

    # Define a list to hold the computed statistics
    output_list = []


    flesch_reading_ease = textstat.flesch_reading_ease(all_text)
    flesch_reading_ease = cap_the_values(flesch_reading_ease, -10, 120)
    check_value(flesch_reading_ease, "flesch_reading_ease")

    flesch_kincaid_grade = textstat.flesch_kincaid_grade(all_text)
    flesch_kincaid_grade = cap_the_values(flesch_kincaid_grade, 0, 15)
    check_value(flesch_kincaid_grade, "flesch_kincaid_grade")

    smog_index = textstat.smog_index(all_text)
    smog_index = cap_the_values(smog_index, 0, 20)
    check_value(smog_index, "smog_index")

    coleman_liau_index = textstat.coleman_liau_index(all_text)
    coleman_liau_index = cap_the_values(coleman_liau_index, 0, 25)
    check_value(coleman_liau_index, "coleman_liau_index")

    automated_readability_index = textstat.automated_readability_index(all_text)
    automated_readability_index = cap_the_values(automated_readability_index, 0, 15)
    check_value(automated_readability_index, "automated_readability_index")

    dale_chall_readability_score = textstat.dale_chall_readability_score(all_text)
    dale_chall_readability_score = cap_the_values(dale_chall_readability_score, 0, 10)
    check_value(dale_chall_readability_score, "dale_chall_readability_score")

    print("Number of words: ", len(all_text.split()))
    difficult_words = textstat.difficult_words(all_text)*100/len(all_text.split())
    check_value(difficult_words, "difficult_words")

    linsear_write_formula = textstat.linsear_write_formula(all_text)
    linsear_write_formula = cap_the_values(linsear_write_formula, 0, 25)
    check_value(linsear_write_formula, "linsear_write_formula")

    gunning_fog = textstat.gunning_fog(all_text)
    gunning_fog = cap_the_values(gunning_fog, 0, 25)
    check_value(gunning_fog, "gunning_fog")

    text_standard = textstat.text_standard(all_text)

    # Extract the numerical values from the strings and convert them to integers
    lower_text_standard = int(text_standard.split(" ")[0].strip("thrdns"))
    higher_text_standard = int(text_standard.split(" and ")[1].split(" ")[0].strip("thrdns"))


    lower_text_standard = cap_the_values(lower_text_standard, 0, 25)
    higher_text_standard = cap_the_values(higher_text_standard, 0, 25)

    check_value(lower_text_standard, "lower_text_standard")
    check_value(higher_text_standard, "higher_text_standard")



    

    output_list.append({
        
        "feature": "Flesch Reading Ease",
        "value": flesch_reading_ease,
        "description": "A readability score that rates text on a 100-point scale. Higher scores indicate easier readability.",
        "category": "Textstat measures"
    })

    output_list.append({
        
        "feature": "Flesch-Kincaid Grade Level",
        "value": flesch_kincaid_grade,
        "description": "A grade level score based on the average number of syllables per word and the average number of words per sentence.",
        "category": "Textstat measures"
    })

    output_list.append({
        
        "feature": "SMOG Index",
        "value": smog_index,
        "description": "A readability index that measures the level of education required to understand a piece of text.",
        "category": "Textstat measures"
    })

    output_list.append({
        
        "feature": "Coleman-Liau Index",
        "value": coleman_liau_index,
        "description": "A readability formula that uses the number of characters, words, and sentences in a text to calculate a grade level score.",
        "category": "Textstat measures"
    })

    output_list.append({
        
        "feature": "Automated Readability Index",
        "value": automated_readability_index,
        "description": "A readability formula that uses the number of characters, words, and sentences in a text to calculate a grade level score.",
        "category": "Textstat measures"
    })

    output_list.append({
        
        "feature": "Dale-Chall Readability Score",
        "value": dale_chall_readability_score,
        "description": "A readability formula that takes into account the number of difficult words in a text to calculate a grade level score.",
        "category": "Textstat measures"
    })

    output_list.append({
        
        "feature": "Number of Difficult Words",
        "value": difficult_words,
        "description": "The number of difficult words in a text, as determined by the Dale-Chall Readability Score formula.",
        "category": "Textstat measures"
    })

    output_list.append({
        
        "feature": "Linsear Write Formula",
        "value": linsear_write_formula,
        "description": "A readability formula that uses the number of easy and difficult words in a text to calculate a grade level score.",
        "category": "Textstat measures"
    })

    output_list.append({
        
        "feature": "Gunning Fog Index",
        "value": gunning_fog,
        "description": "A readability index that estimates the years of formal education needed to understand a piece of text.",
        "category": "Textstat measures"
    })


    output_list.append({
        
        "feature": "Lower Text Standard",
        "value": lower_text_standard,
        "description": "The standardized level of text complexity based on the Common Core State Standards.",
        "category": "Textstat measures"
    })


    output_list.append({
        
        "feature": "Higher Text Standard",
        "value": higher_text_standard,
        "description": "The standardized level of text complexity based on the Common Core State Standards.",
        "category": "Textstat measures"
    })

   

    output_list.append({
        
        "feature": "Textstat measures Time",
        "value": time.time() - start_time,
        "description": "The time it took to calculate the textstat measures.",
        "category": "Textstat measures"
    })
    

    return output_list