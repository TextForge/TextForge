# TextForge

How to install TextForge?:

pip install git+https://github.com/TextForge/TextForge.git

How to use TextForge?


from TextForge import textForge

try:
    features = pd.read_csv('features_mini.csv')
except:
    features = pd.DataFrame(columns=['dataset'])

#list all the files in TextForge/Train
files = os.listdir('Train_Data_Folder')

print(files)

#for each item in files add 'TextForge/Train' to the beginning
files = ['Train_Data_Folder/' + file for file in files]

#keep only the files that are not in features.csv
files = [file for file in files if file not in features['dataset'].values]

for file in files:
    print("Running:", file)
    try:
        df = pd.read_csv(file)
        #remove null values
        df = df.dropna()
        print(df)
        f = textForge.extract_features(df, file, pd.DataFrame(), config)
        features = pd.concat([features, f], ignore_index=True)
        features.to_csv('features_mini.csv', index=False)
    except:
        print('error', file)
        pass

