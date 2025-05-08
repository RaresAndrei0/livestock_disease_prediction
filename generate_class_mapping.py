import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

# Load the original dataset that contains the 'Disease' column (not the encoded one)
df = pd.read_csv("dataset/animal_disease_dataset.csv")  # Make sure this file has 'Disease' column

# Fit LabelEncoder on the original disease names
label_encoder = LabelEncoder()
label_encoder.fit(df["Disease"])

# Map class indices to disease names
class_mapping = {index: name for index, name in enumerate(label_encoder.classes_)}

# Save to JSON
with open("disease_classes.json", "w") as f:
    json.dump(class_mapping, f, indent=4)

print("✅ disease_classes.json file has been created with actual disease names.")