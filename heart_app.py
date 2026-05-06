import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# Título
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Enter the patient's clinical data to predict cardiovascular disease risk.")
st.divider()

# Cargar y entrenar modelo
@st.cache_resource
def cargar_modelo():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columnas = ['edad', 'sexo', 'tipo_dolor_pecho', 'presion_reposo',
                'colesterol', 'azucar_ayunas', 'ecg_reposo', 'fc_maxima',
                'angina_ejercicio', 'depresion_st', 'pendiente_st',
                'vasos_coloreados', 'talasemia', 'enfermedad']
    df = pd.read_csv(url, names=columnas, na_values='?')
    df['enfermedad'] = (df['enfermedad'] > 0).astype(int)
    df = df.dropna()
    X = df.drop('enfermedad', axis=1)
    y = df['enfermedad']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    modelo = SVC(probability=True, random_state=42)
    modelo.fit(X_scaled, y)
    return modelo, scaler

modelo, scaler = cargar_modelo()

# Formulario
col1, col2 = st.columns(2)

with col1:
    edad = st.slider("Age", 20, 80, 50)
    sexo = st.selectbox("Sex", ["Female", "Male"])
    presion = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 120)
    colesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 250)
    fc_max = st.slider("Maximum Heart Rate", 70, 210, 150)
    angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

with col2:
    dolor = st.selectbox("Chest Pain Type", 
                         ["Typical Angina", "Atypical Angina", 
                          "Non-anginal Pain", "Asymptomatic"])
    azucar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    ecg = st.selectbox("Resting ECG", 
                       ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    depresion = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
    pendiente = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
    vasos = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    talasemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

st.divider()

# Convertir valores
sexo_val = 1 if sexo == "Male" else 0
dolor_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(dolor) + 1
azucar_val = 1 if azucar == "Yes" else 0
ecg_val = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(ecg)
angina_val = 1 if angina == "Yes" else 0
pendiente_val = ["Upsloping", "Flat", "Downsloping"].index(pendiente) + 1
talasemia_val = [3, 6, 7][["Normal", "Fixed Defect", "Reversible Defect"].index(talasemia)]

# Predicción
if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
    paciente = np.array([[edad, sexo_val, dolor_val, presion, colesterol,
                          azucar_val, ecg_val, fc_max, angina_val, depresion,
                          pendiente_val, vasos, talasemia_val]])
    paciente_scaled = scaler.transform(paciente)
    probabilidad = modelo.predict_proba(paciente_scaled)[0][1]

    st.divider()
    if probabilidad > 0.5:
        st.error(f"### 🔴 HIGH RISK — {probabilidad:.1%} probability of heart disease")
        st.warning("⚠️ This patient shows high cardiovascular risk. Please consult a cardiologist.")
    else:
        st.success(f"### 🟢 LOW RISK — {probabilidad:.1%} probability of heart disease")
        st.info("✅ This patient shows low cardiovascular risk based on the provided data.")

    st.caption("⚠️ This tool is for educational purposes only and does not replace medical diagnosis.")