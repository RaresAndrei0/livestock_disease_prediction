import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

# Load the trained model
model = load_model("model/disease_model.h5")

# Load training history
with open("model/training_history.pkl", "rb") as f:
    history = pickle.load(f)

# Load the dataset
df_encoded = pd.read_csv("dataset/processed_df.csv")

# Visualization 1: Distribution of disease labels
plt.figure(figsize=(8, 6))
sns.countplot(x='Disease_label', data=df_encoded)
plt.title('Distribution of Disease Classes')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig("visualizations/Distribution_diseases.png")

# Visualization 2: Training and validation accuracy/loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("visualizations/Scores.png")
