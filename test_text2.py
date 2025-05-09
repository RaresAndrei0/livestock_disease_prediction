import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json

# Load model
model = load_model("model/disease_model.h5")

# Load processed DataFrame
df_encoded = pd.read_csv("dataset/processed_df.csv")
X = df_encoded.drop(columns=["Disease_label"])

# Fit scaler
scaler = StandardScaler()
scaler.fit(X)

# Load class map
with open("disease_classes.json") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

# Get column names
all_features = list(X.columns)

while True:
    print("\n Testare model AI cu date introduse manual:")
    try:
        age = float(input(" Introduceți vârsta animalului: "))
        temp = float(input(" Introduceți temperatura (Fahrenheit): "))
        animal = input(" Alegeți animalul (cow, goat, sheep, buffalo): ").strip().lower()
        symptoms = input(" Introduceți simptome separate prin virgulă: ").strip().lower().split(",")

        symptoms = [s.strip() for s in symptoms]
        animal_col = f"Animal_{animal}"

        if animal_col not in all_features:
            print(f" Animal invalid: {animal}")
            continue

        # Build input vector
        input_dict = {col: 0 for col in all_features}
        input_dict["Age"] = age
        input_dict["Temperature"] = temp
        input_dict[animal_col] = 1

        for sym_col in all_features:
            for s in symptoms:
                if s and s in sym_col.lower():
                    input_dict[sym_col] = 1

        input_vector = [input_dict[col] for col in all_features]
        input_scaled = scaler.transform([input_vector])
        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]

        print(f"\nPredicție: {label_map[predicted_class]}")

    except Exception as e:
        print(f"Eroare: {e}")