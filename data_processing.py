import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def load_and_process_data():
    df = pd.read_csv('livestock_disease.csv')
    df['Symptoms'] = df['Symptoms'].str.lower().str.split(',')

    mlb = MultiLabelBinarizer()
    symptoms_encoded = pd.DataFrame(mlb.fit_transform(df['Symptoms']),columns=mlb.classes_)
