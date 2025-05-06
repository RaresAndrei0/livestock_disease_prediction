import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def load_and_process_data():
    df = pd.read_csv('livestock_disease.csv')
    df['Symptoms'] = df['Symptoms'].str.lower().str.split(',')

    mlb = MultiLabelBinarizer()
    symptoms_encoded = pd.DataFrame(mlb.fit_transform(df['Symptoms']),columns=mlb.classes_)
    enc = LabelEncoder()

    df['Severity'] = enc.fit_transform(df['Severity'])
    df['Curability'] = enc.fit_transform(df['Curability'])
    df['Affected Animals'] = df['Affected Animals'].astype('category').cat.codes
    df['Duration'] = df['Duration'].astype('category').cat.codes

    df['Disease'] = df['Disease'].astype('category')
    y = df['Disease'].cat.codes
    disease_classes = dict(enumerate(df['Disease'].cat.categories))

    X = pd.concat([symptoms_encoded, df[['Severity', 'Curability', 'Affected Animals', 'Duration']]], axis=1)

    return X, y, disease_classes
