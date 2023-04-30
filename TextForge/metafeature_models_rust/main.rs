use std::collections::HashMap;

use pandas::DataFrame;
use regex::Regex;

fn convert_classification_dataset_to_features(dataframe: &DataFrame) -> HashMap<String, usize> {
    let mut df = dataframe.clone();

    // Remove rows with any column having null values
    df = df.dropna(None, "all", None);

    let mut features_obj = HashMap::new();

    features_obj.insert("size".to_owned(), df.shape().unwrap()[0]);
    features_obj.insert("columns".to_owned(), df.shape().unwrap()[1]);
    features_obj.insert("unique_labels".to_owned(), df["label"].nunique().unwrap());
    features_obj.insert("null_values".to_owned(), df.isnull().sum().sum());

    features_obj
}
