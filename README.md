# TextForge README

TextForge is a Python package that provides a simple interface for natural language processing (NLP) tasks such as feature extraction, data cleaning, and data preprocessing. This README file provides instructions on how to use TextForge on GitHub.

## Installation

To install TextForge, follow these steps:

1. Install the package using pip by running the following command in your terminal:

```
pip install git+https://github.com/TextForge/TextForge.git
```

## Usage

To use TextForge, follow these steps:

1. Import TextForge into your Python script using the following code:

```python
from TextForge import textForge
```

2. Create a Pandas DataFrame to store the extracted features. You can load an existing DataFrame from a CSV file using the following code:

```python
try:
    features = pd.read_csv('features.csv')
except:
    features = pd.DataFrame(columns=['dataset'])
```

3. List all the files in the `Train_Data_Folder` directory using the following code:

```python
files = os.listdir('Train_Data_Folder')
```

4. For each item in `files`, add `'Train_Data_Folder/'` to the beginning of the file path using the following code:

```python
files = ['Train_Data_Folder/' + file for file in files]
```

5. Keep only the files that are not in the `features.csv` file using the following code:

```python
files = [file for file in files if file not in features['dataset'].values]
```

6. For each file in `files`, extract the features using TextForge and store the results in the `features` DataFrame using the following code:

```python
for file in files:
    print("Running:", file)
    try:
        df = pd.read_csv(file)
        #remove null values
        df = df.dropna()
        print(df)
        f = textForge.extract_features(df, file, pd.DataFrame(), config)
        features = pd.concat([features, f], ignore_index=True)
        features.to_csv('features.csv', index=False)
    except:
        print('error', file)
        pass
```

7. Save the `features` DataFrame to a CSV file using the following code:

```python
features.to_csv('features.csv', index=False)
```

## Conclusion

You should now be able to use TextForge on GitHub by following these instructions. If you have any questions or encounter any issues, please refer to the documentation or open an issue on the GitHub repository.

See sample_code.ipynb for a sample on how to use this package