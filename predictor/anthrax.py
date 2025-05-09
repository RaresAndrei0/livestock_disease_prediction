import pandas as pd

df = pd.read_csv("dataset/feature_engineered_dataset.csv")
anthrax_samples = df[df["Disease_label"] == 0] 
mean_symptoms = anthrax_samples.mean(numeric_only=True)


important_symptoms = mean_symptoms[mean_symptoms > 0.5]
print("Simptome comune pentru anthrax:")
print(important_symptoms)