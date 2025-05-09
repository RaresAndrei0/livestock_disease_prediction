import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("dataset/animal_disease_dataset.csv")  


label_encoder = LabelEncoder()
label_encoder.fit(df["Disease"])


class_mapping = {index: name for index, name in enumerate(label_encoder.classes_)}


with open("disease_classes.json", "w") as f:
    json.dump(class_mapping, f, indent=4)

print("disease_classes.json file has been created with actual disease names.")