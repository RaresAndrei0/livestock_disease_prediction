import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load model
model = load_model("model/disease_model.h5")

# Load the processed dataset for reference (structure)
df_encoded = pd.read_csv("dataset/processed_df.csv")

# Separate features and target
X = df_encoded.drop(columns=['Disease_label'])
y = df_encoded['Disease_label']

# Fit the scaler on the original dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Get column names for manual input
feature_columns = X.columns

# Manual testing loop
while True:
    print("\nEnter symptoms/features as binary values (1/0) for each of the following:")
    print("Type 'exit' to quit.")

    user_input = []
    for feature in feature_columns:
        val = input(f"{feature}: ").strip()
        if val.lower() == 'exit':
            exit()
        try:
            user_input.append(int(val))
        except:
            print("Invalid input. Please enter 1 or 0.")
            break

    if len(user_input) != len(feature_columns):
        print("⚠️ Input length mismatch. Try again.")
        continue

    # Preprocess input
    user_input_scaled = scaler.transform([user_input])

    # Predict
    prediction = model.predict(user_input_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]

    print(f"\n🩺 Predicted disease class: {predicted_class}")