import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json

# Load model
model = load_model("model/disease_model.h5")

# Load dataset for column structure & fit scaler
df = pd.read_csv("dataset/feature_engineered_dataset.csv")
X = df.drop(columns=["Disease_label"])
scaler = StandardScaler()
scaler.fit(X)

# Load class map (disease index to name)
with open("disease_classes.json") as f:
    class_map = json.load(f)
inv_class_map = {v: k for k, v in class_map.items()}

# Start interactive input
print("\n🔥 Testare model AI cu date introduse manual:")

# Input age, temperature, animal
age = float(input("🧬 Introduceți vârsta animalului: "))
temperature = float(input("🌡️  Introduceți temperatura (Fahrenheit): "))
animal = input("🐄 Alegeți animalul (cow, goat, sheep, buffalo): ").strip().lower()

# Input simptome
symptoms = input("💉 Introduceți simptome separate prin virgulă: ").strip().lower().split(",")

# Curățare spații
symptoms = [s.strip() for s in symptoms if s.strip()]

# Inițializează un rând cu zero pe toate coloanele
input_row = pd.DataFrame(data=[np.zeros(X.shape[1])], columns=X.columns)

# Setăm valorile de bază
input_row["Age"] = age
input_row["Temperature"] = temperature

# Setăm animalul
animal_col = f"Animal_{animal}"
if animal_col in input_row.columns:
    input_row[animal_col] = 1
else:
    print(f"⚠️ Animal invalid: {animal}")
    exit()

# Setăm simptomele (în coloanele Symptom 1/2/3_)
symptom_cols = X.columns
for i, symptom in enumerate(symptoms):
    for col in symptom_cols:
        if col.endswith(symptom) and col.startswith(f"Symptom {i+1}_"):
            input_row[col] = 1
            break

# Normalizează și prezice
input_scaled = scaler.transform(input_row)
prediction = model.predict(input_scaled)
predicted_class = np.argmax(prediction)

# Afișare rezultat
print(f"\n✅ Predicție: {inv_class_map[str(predicted_class)]}")