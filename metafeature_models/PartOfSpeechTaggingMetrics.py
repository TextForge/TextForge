from collections import Counter
import nltk
import time

def get_part_of_speech_tagging_metrics(dataframe):

    start_time = time.time()

    all_text = " ".join(dataframe["text"].map(str))
    all_words = all_text.split()
    tagged_words = nltk.pos_tag(all_words, tagset='universal')
    tag_counts = Counter(tag for word, tag in tagged_words)
    total_tagged = sum(tag_counts.values())
    tag_pct = {word: count/total_tagged for word, count in tag_counts.items()}

    output_list = [
        {
            "feature": f"POS Tagging - {tag}",
            "value": tag_pct.get(tag, 0),
            "description": f"The percentage of words in the text that are {tag.lower()}.",
            "category": "POS tagging statistics / lexical features"
        } for i, tag in enumerate(['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X'], 1)
    ]

    output_list.append({
        
        "feature": "POS Tagging Time",
        "value": time.time() - start_time,
        "description": "The time taken to calculate the POS tagging features.",
        "category": "POS tagging statistics / lexical features"
    })

    return output_list
