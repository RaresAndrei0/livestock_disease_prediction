import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json

# Load model
model = load_model("model/disease_model.h5")

# Load processed DataFrame (with full symptom feature columns)
df_encoded = pd.read_csv("dataset/processed_df.csv")
X = df_encoded.drop(columns=["Disease_label"])

# Fit scaler to original training data
scaler = StandardScaler()
scaler.fit(X)

# Get list of all symptoms/features
all_symptoms = set(X.columns)

# Load disease class map (optional)
try:
    with open("disease_classes.json") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
except:
    label_map = None

# Input loop
while True:
    print("\nEnter symptoms separated by commas (e.g., 'fever, cough') or type 'exit':")
    user_input = input(">> ").strip().lower()
    if user_input == "exit":
        break

    input_symptoms = [sym.strip() for sym in user_input.split(",")]

    # Build one-hot input vector
    input_vector = [1 if symptom in input_symptoms else 0 for symptom in X.columns]

    # Scale input
    input_scaled = scaler.transform([input_vector])

    # Predict
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]

    if label_map:
        predicted_name = label_map[predicted_class]
        print(f"🩺 Predicted disease: {predicted_name}")
    else:
        print(f"🩺 Predicted disease class index: {predicted_class}")