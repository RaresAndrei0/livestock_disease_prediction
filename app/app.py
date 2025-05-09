from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__, template_folder='templates')

# Încarcă resusele la pornire
model = load_model("model/disease_model.h5")
df_encoded = pd.read_csv("dataset/processed_df.csv")
X = df_encoded.drop(columns=["Disease_label"])
scaler = StandardScaler().fit(X)

with open("disease_classes.json") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}
all_features = list(X.columns)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Construiește vectorul de input
        input_dict = {col: 0 for col in all_features}
        input_dict["Age"] = data['age']
        input_dict["Temperature"] = data['temperature']
        
        animal_col = f"Animal_{data['animal']}"
        if animal_col not in all_features:
            return jsonify({"error": "Animal invalid"}), 400
        input_dict[animal_col] = 1

        # Procesează simptomele
        for sym_col in all_features:
            for s in data['symptoms']:
                if s and s in sym_col.lower():
                    input_dict[sym_col] = 1

        input_vector = [input_dict[col] for col in all_features]
        input_scaled = scaler.transform([input_vector])
        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return jsonify({
            "prediction": label_map.get(predicted_class, "Unknown disease")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)