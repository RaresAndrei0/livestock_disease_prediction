import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

# 1. Încarcă datele
df = pd.read_csv("livestock_diseases_150.csv")  # ← asigură-te că fișierul există

# 2. Normalizează textul
df["Symptoms"] = df["Symptoms"].astype(str).apply(lambda x: x.lower())

# 3. Elimină bolile cu mai puțin de 2 exemple
value_counts = df["Disease"].value_counts()
df = df[df["Disease"].isin(value_counts[value_counts > 1].index)].reset_index(drop=True)

# 4. Tokenizator custom (separă după virgulă)
def custom_tokenizer(text):
    return text.split(",")

# 5. Vectorizează simptomele
vectorizer = CountVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
X_symptoms = vectorizer.fit_transform(df["Symptoms"])

# 6. Encode bolile
label_encoder = LabelEncoder()
diseases_encoded = label_encoder.fit_transform(df["Disease"])

# 7. Împarte în set de antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(
    X_symptoms, diseases_encoded, test_size=0.2, stratify=diseases_encoded, random_state=42
)

# 8. Creează și antrenează modelul
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Evaluează modelul
y_pred = model.predict(X_test)

print("=== EVALUARE MODEL ===")
print("Acuratețe:", accuracy_score(y_test, y_pred))

labels_in_test = unique_labels(y_test, y_pred)
target_names = label_encoder.inverse_transform(labels_in_test)
print("Raport de clasificare:\n", classification_report(y_test, y_pred, labels=labels_in_test, target_names=target_names))

# 10. Salvează modelul și encoderele
joblib.dump(model, "trained_model.pkl")
joblib.dump(vectorizer, "symptom_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

with open("disease_classes.json", "w") as f:
    json.dump({i: c for i, c in enumerate(label_encoder.classes_)}, f)

print("✅ Modelul și fișierele auxiliare au fost salvate.")
