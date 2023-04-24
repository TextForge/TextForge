import numpy as np

from metafeature_models import (
    AutoMLFeatures,
    CategoryDocumentFrequencyFeatures,
    InitialFeatures,
    HardnessFeatures,
    LandmarkingMetrics,
    MUDOF,
    PartOfSpeechTaggingMetrics,
    PrincipalComponentAnalysisFeatures,
    TextFeatureTaxonomyMetrics,
    TextStatMetrics,
    VocabularyMetrics,
    DocumentLengthMetrics,
)

import pandas as pd
import time

def extract_features(file_name, current_features):
    dataframe = pd.read_csv(file_name).dropna()
    dataframe = dataframe.groupby('label').filter(lambda x: len(x) > 10)
    dataframe['text'] = dataframe['text'].str.lower()

    # Create an empty list to store the output features
    output_list = []

    features = [
        (InitialFeatures.get_initial_features, 'Calculating initial features...'),
        (TextStatMetrics.get_textstat_metrics, 'Calculating TextStat metrics...'),
        (VocabularyMetrics.get_vocabulary_metrics, 'Calculating vocabulary metrics...'),
        (PartOfSpeechTaggingMetrics.get_part_of_speech_tagging_metrics, 'Calculating part-of-speech tagging metrics...'),
        (CategoryDocumentFrequencyFeatures.get_category_document_frequency_features, 'Calculating category document frequency meta-features...'),
        (DocumentLengthMetrics.get_words_per_doc_features, 'Calculating document length meta-features...'),
        (PrincipalComponentAnalysisFeatures.get_principal_component_analysis_metafeatures, 'Calculating principal component analysis meta-features...'),
        (HardnessFeatures.get_hardness_features, 'Calculating hardness features...'),
        (LandmarkingMetrics.get_landmarking_metrics, 'Calculating landmarking meta-features...'),
        (MUDOF.get_MUDOF_features, 'Calculating multi-class discrepancy features...'),
        (TextFeatureTaxonomyMetrics.get_text_feature_taxonomy_metrics, 'Calculating text feature taxonomy metrics...'),
        (AutoMLFeatures.get_automl_metafeatures, 'Calculating AutoML features...'),
    ]

    for function, message in features:
        print(message)
        start_time = time.time()
        for item in function(dataframe):
            output_list.append(item)
        end_time = time.time()
        print(f'Time taken: {round((end_time - start_time),0)} seconds')

    # Convert the list of dictionaries to a Pandas DataFrame
    output_df = pd.DataFrame(output_list)


    # # Explode the 'value' column into multiple rows in the cases where the value is a list
    # output_df = output_df.explode('value')

    # # Create a new column to store the unique suffix for each key
    # output_df['suffix'] = "_" + str(output_df.groupby('feature').cumcount())

    # # Add the suffix to the key column to create the new key column
    # output_df['feature'] = output_df['feature'] + output_df['suffix'].astype(str)

    # output_df['feature'] = output_df['feature'].str.replace('_0', '')





    # return output_df

    # Drop the old key and suffix columns and keep only the feature and value columns
    output_df = output_df[['feature', 'value']]

    # Reset the index of the DataFrame
    output_df = output_df.reset_index(drop=True)

    # Create a new dataframe with 1 row
    new_df = pd.DataFrame(columns=output_df['feature'].unique(), index=[0])

    # Loop through each row in the original dataframe and add values to the new dataframe
    for _, row in output_df.iterrows():
        feature_name = row['feature']
        value = row['value']
        new_df.loc[0, feature_name] = value

    # Add a 'dataset' column and move it to the first column
    new_df['dataset'] = file_name
    new_df = new_df[['dataset'] + list(new_df.columns[:-1])]

    # Return the new dataframe
    return new_df