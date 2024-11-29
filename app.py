import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load datasets
@st.cache_data
def load_data():
    df = pd.read_csv("Training.csv")
    tr = pd.read_csv("Testing.csv")
    
    # Encode diseases
    disease_dict = {
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

    df.replace({'prognosis': disease_dict}, inplace=True)
    df = df.infer_objects(copy=False)

    tr.replace({'prognosis': disease_dict}, inplace=True)
    tr = tr.infer_objects(copy=False)
    
    return df, tr, disease_dict

try:
    df, tr, disease_dict = load_data()
except FileNotFoundError as e:
    st.error("Data files not found. Please ensure `Training.csv` and `Testing.csv` are available.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading data: {e}")
    st.stop()

l1 = list(df.columns[:-1])  
X = df[l1]
y = df['prognosis']
X_test = tr[l1]
y_test = tr['prognosis']

# trained models
@st.cache_resource
def train_models():
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB()
    }
    trained_models = {}
    for model_name, model_obj in models.items():
        model_obj.fit(X, y)
        acc = accuracy_score(y_test, model_obj.predict(X_test))
        trained_models[model_name] = (model_obj, acc)
    return trained_models

trained_models = train_models()

def predict_disease(model, symptoms):
    input_test = np.zeros(len(l1))
    for symptom in symptoms:
        if symptom in l1:
            input_test[l1.index(symptom)] = 1
    prediction = model.predict([input_test])[0]
    return list(disease_dict.keys())[list(disease_dict.values()).index(prediction)]

# Streamlit interface
st.title("Disease Predictor Using Machine Learning")

# Input fields
st.subheader("Enter Patient Details")
st.text("For accurate results, please select at least 3 symptoms.")
name = st.text_input("Name of Patient")

if not name.strip():
    st.warning("Please enter the patient's name.")

symptom1 = st.selectbox("Symptom 1", ["None"] + l1)
symptom2 = st.selectbox("Symptom 2", ["None"] + l1)
symptom3 = st.selectbox("Symptom 3", ["None"] + l1)
symptom4 = st.selectbox("Symptom 4", ["None"] + l1)
symptom5 = st.selectbox("Symptom 5", ["None"] + l1)

symptoms_selected = [s for s in [symptom1, symptom2, symptom3, symptom4, symptom5] if s != "None"]

if st.button("Predict Disease"):
    if len(symptoms_selected) < 3:
        st.error("Please select at least 3 symptoms for accurate prediction.")
    else:
        for model_name, (model, acc) in trained_models.items():
            prediction = predict_disease(model, symptoms_selected)
            st.subheader(f"{model_name} Prediction:")
            st.write(f"Predicted Disease: **{prediction}**")
            st.write(f"Model Accuracy: **{acc * 100:.2f}%**")

# Final Note
st.markdown("---")
st.warning("**Caution:** This system is designed for informational purposes only. Please visit a healthcare provider for any medical concerns.")
