from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from data_processing import load_and_process_data

# 1. Încarcă și preprocesează datele
X, y, disease_classes = load_and_process_data()

# 2. Împarte în seturi de antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Creează modelul Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Antrenează modelul pe datele de training
model.fit(X_train, y_train)

# 5. Testează modelul pe datele de test
y_pred = model.predict(X_test)

# 6. Afișează scorul de acuratețe și raportul complet
print("=== EVALUARE MODEL ===")
print("Acuratețe:", accuracy_score(y_test, y_pred))
print("Raport de clasificare:\n", classification_report(
    y_test,
    y_pred,
    labels=list(disease_classes.keys()),
    target_names=list(disease_classes.values()),
    zero_division=0
))

# 7. Salvează modelul antrenat în fișier
joblib.dump(model, "trained_model.pkl")
print("✅ Model salvat ca 'trained_model.pkl'")

# 8. Salvează mapping-ul între coduri și denumiri de boli (pentru Django)
import json
with open("disease_classes.json", "w") as f:
    json.dump(disease_classes, f)
print("✅ Mapping salvat în 'disease_classes.json'")
