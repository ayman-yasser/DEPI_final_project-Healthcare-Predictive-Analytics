#app.py
import streamlit as st
import mlflow
import pandas as pd

import joblib

# Load the model from MLflow
logged_model = 'runs:/e2b2ddd9b9694e9faf2cc30f089a544a/DecisionTreeClassifier/with-class-weights'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Define the required columns for the input data
columns = ['mental_health_score','exercise_per_week','age','bmi','liver_function','blood_sugar','hospital_stay_days','hospital_visits','medication','smoker','diabetes','diagnosis']

# User interface setup
st.title("💊 Treatment Outcome Prediction")

# Sidebar ABOUT the app
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This app uses a machine learning model to predict the outcome of a treatment based on user inputs.
                    
**-Model**: Decision Tree  
**-Technique**: SMOTE for class balancing  
**-Platform**: Streamlit + MLflow  
""")

 #CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.5em 2em;
        }
    </style>
""", unsafe_allow_html=True)

# Add user inputs for the features
mental_health_score = st.number_input('🧠Mental Health Score', min_value=0.0, max_value=10.0, value=4.5, step=0.1)
medication = st.selectbox('💊Medication', ['No Drug','Lisinopril', 'Statins', 'Metformin', 'Beta Blockers'], index=0)
exercise_per_week = st.number_input('🏃‍♀️Exercise per Week ', min_value=0.0, max_value=7.0, value=3.0, step=1.0)
age = st.number_input('🎂Age', min_value=0.0, max_value=100.0, value=45.0, step=1.0)
bmi = st.number_input('⚖️BMI', min_value=10.0, max_value=50.0, value=29.8, step=0.1)
liver_function = st.number_input('🧪Liver Function', min_value=0.0, max_value=100.0, value=6.3, step=0.1)
blood_sugar = st.number_input('🩸Blood Sugar Level', min_value=50.0, max_value=200.0, value=90.5, step=0.1)
smoker = st.selectbox('🚬Smoker', ['No', 'Yes'], index=0)
diabetes = st.selectbox('🍬Diabetes', ['No', 'Yes'], index=0)
diagnosis = st.selectbox('🩺Diagnosis', ['Liver Disease', 'Healthy', 'Kidney Disease', 'Heart Disease', 'Diabetes', 'Hypertension'], index=0)
hospital_stay_days = st.number_input('🏥Hospital Stay Days', min_value=0.0, max_value=100.0, value=3.0, step=1.0)
hospital_visits = st.number_input('📅Hospital Visits', min_value=0.0, max_value=100.0, value=4.0, step=1.0)

# Convert categorical values to numerical values
# Map Medication to an integer value
medication_mapping = {
    'No Drug': 3,
    'Lisinopril': 1,
    'Statins': 4,
    'Metformin': 2,
    'Beta Blockers': 0
}
# Map Diagnosis to an integer value
diagnosis_mapping = {
    'Liver Disease': 5,
    'Healthy': 1,
    'Kidney Disease': 4,
    'Heart Disease': 2,
    'Diabetes': 0,
    'Hypertension': 3
}
# Convert selected Medication and Diagnosis to integers
medication_value = int(medication_mapping[medication])
diagnosis_value = int(diagnosis_mapping[diagnosis])

# Convert Smoker and Diabetes to binary (1 or 0)
smoker = 1 if smoker == 'Yes' else 0
diabetes = 1 if diabetes == 'Yes' else 0

# Prepare the input data as a DataFrame
input_data = pd.DataFrame([[
    mental_health_score, exercise_per_week, age, bmi,
    liver_function, blood_sugar,
    hospital_stay_days, hospital_visits, medication_value, smoker, diabetes, diagnosis_value
]], columns=columns)

scaler = joblib.load('scaler.pkl')

numeric_cols = columns[:8]
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
print(input_data)
# When the 'Predict Treatment Outcome' button is clicked
if st.button('🎯 Predict Treatment Outcome'):
    # Make the prediction using the model
    prediction = loaded_model.predict(input_data)

    outcome_mapping = {
        0: '❌ Failure ',
        1: '⚠️ Improvement ',
        2: '✅ Recovered '
    }

    # Display the result
    st.write(f'Predicted Treatment Outcome: {outcome_mapping[prediction[0]]}')
    