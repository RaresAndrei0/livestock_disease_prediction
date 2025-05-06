# data_processing.py temporar - doar pentru regenerare
import pandas as pd

data = {
    'Disease': [
        'Foot and Mouth Disease', 'Bovine Tuberculosis', 'Brucellosis', 'Avian Influenza', 'Bluetongue', 'Mantitis','African Swine Fever', 'Salmonellosis',
        'Johnes Disease', 'Rabies', 'Lumpy Skin Disease', 'Newcastle Disease', 'Fowlpox', 'Leptospirosis', 'Anthrax'
        ],

    'Symptoms': [
        'Fever, Blisters on mouth and feet, Lameness, Loss of appetite', 'Coughing, Weight Loss, Fever, Swollen lymph nodes', 'Fever, Joint pain, Abortion, Weak Calves',
        'Fever, Coughing, Respiratory distress, Death in birds', 'Fever, Swelling, Inflamation, Lameness', 'Swelling of udder, Heat, Pain, Reduced milk production', 'High fever, Red skin lesions, Internal bleeding, Death in pigs',
        'Fever, diarrhea, Abdominal pain, Vomiting', 'Weight loss, Diarrhea, Reduced milk production, Arthritis', 'Behavioral changes, Drooling, Paralysis, Death', 'Swelling of skin, Fever, Lesions, Reduced milk production',
        'Respiratory distress, Couching, Nasal Dsicharge, Paralysis', 'Fever, Skin lesions, Swollen eyes, Reduced egg production', 'Fever, Lethargy, Yellowing of eyes, Vomiting', 'Fever, Sudden death, Swelling, Blackening of tissue'
        ],

    'Affected Animals': [
        'Cows, Sheep, Goats', 'Cows', 'Cows, Pigs', 'Chickens, Ducks', 'Cows, Sheep', 'Cows', 'Pigs', 'Pigs, Cows',
        'Cows, Sheep', 'Dogs, Cows, Horses', 'Cows', 'Chickens, Ducks, Turkeys', 'Chickens, Ducks', 'Cows, Goats', 'Pigs, Cattle'
        ],

    'Severity': [
        'High', 'High', 'Moderate', 'High', 'Moderate', 'Low', 'High', 'Moderate', 'High', 'High',
        'Moderate', 'Moderate', 'Moderate', 'High', 'High'
        ],

    'Duration':[
        '1-2 weeks', 'Months', 'Chronic', '1-3 weeks', '3-4 weeks', '1-2 weeks', 'Acute', '2-3 days', 'Chronic',
        '7-10 days', '2-3 weeks', '1-2 weeks', '1-2 weeks', '1-2 weeks', 'Acute'
        ],

    'Curability':[
        'Incurable', 'Incurable', 'Incurable', 'Incurable', 'Incurable', 'Curable', 'Incurable', 'Curable', 'Incurable', 'Incurable',
        'Curable', 'Incurable', 'Curable', 'Curable', 'Incurable'
    ]
}

df = pd.DataFrame(data)
df.to_csv('livestock_disease.csv', index=False)
print("✅ Fișierul CSV a fost regenerat corect.")
