import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load model and data once
model = load_model("model/disease_model.h5")
df = pd.read_csv("dataset/processed_df.csv")
X = df.drop(columns=["Disease_label"])
scaler = StandardScaler()
scaler.fit(X)
columns = X.columns.tolist()

with open("disease_classes.json") as f:
    class_map = json.load(f)
class_map = {int(k): v for k, v in class_map.items()}

def predict_disease(age, temp, animal, symptoms):
    animal_col = f"Animal_{animal.lower()}"
    if animal_col not in columns:
        raise ValueError(f"Invalid animal: {animal}")

    input_dict = {col: 0 for col in columns}
    input_dict["Age"] = age
    input_dict["Temperature"] = temp
    input_dict[animal_col] = 1

    for feature in columns:
        for sym in symptoms:
            if sym.lower() in feature.lower():
                input_dict[feature] = 1

    input_vector = [input_dict[col] for col in columns]
    input_scaled = scaler.transform([input_vector])
    prediction = model.predict(input_scaled)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    return class_map.get(predicted_class, f"Class {predicted_class}")