import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the datasets
df = pd.read_csv("Training.csv")
tr = pd.read_csv("Testing.csv")

# Encode target labels
disease_mapping = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
    'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
    'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
    'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28,
    'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32,
    'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
    'Psoriasis': 39, 'Impetigo': 40
}

df.replace({'prognosis': disease_mapping}, inplace=True)
tr.replace({'prognosis': disease_mapping}, inplace=True)

# Features and targets
l1 = list(df.columns[:-1])
X = df[l1]
y = df["prognosis"]
X_test = tr[l1]
y_test = tr["prognosis"]

# Models
def train_and_predict(model, symptoms):
    model.fit(X, y)
    input_test = np.array([1 if symptom in symptoms else 0 for symptom in l1]).reshape(1, -1)
    prediction = model.predict(input_test)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return prediction[0], accuracy

# Streamlit UI
st.title("Disease Predictor using Machine Learning")
st.write("Select symptoms and predict the most likely disease.")

# Input selection
symptoms = st.multiselect("Select symptoms:", l1)

# Prediction buttons
if st.button("Predict using Decision Tree"):
    prediction, acc = train_and_predict(DecisionTreeClassifier(), symptoms)
    st.success(f"Predicted Disease: {list(disease_mapping.keys())[prediction]}")
    st.info(f"Model Accuracy: {acc:.2f}")

if st.button("Predict using Random Forest"):
    prediction, acc = train_and_predict(RandomForestClassifier(), symptoms)
    st.success(f"Predicted Disease: {list(disease_mapping.keys())[prediction]}")
    st.info(f"Model Accuracy: {acc:.2f}")

if st.button("Predict using Naive Bayes"):
    prediction, acc = train_and_predict(GaussianNB(), symptoms)
    st.success(f"Predicted Disease: {list(disease_mapping.keys())[prediction]}")
    st.info(f"Model Accuracy: {acc:.2f}")
