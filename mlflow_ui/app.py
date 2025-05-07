import streamlit as st
import mlflow
import pandas as pd

# Load the model from MLflow
logged_model = 'runs:/57e23e3b1c414ee38498ddee565cf5c3/DecisionTreeClassifier/with-SMOTE'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Define the required columns for the input data
columns = [
    'mental_health_score', 'medication', 'exercise_per_week', 'age', 'bmi',
    'liver_function', 'blood_sugar', 'smoker', 'diabetes', 'diagnosis',
    'hospital_stay_days', 'hospital_visits'
]

# User interface setup
st.title("Treatment Outcome Prediction")

# Add user inputs for the features
mental_health_score = st.number_input('Mental Health Score', min_value=0.0, max_value=10.0, value=4.5, step=0.1)
medication = st.selectbox('Medication', ['No Drug','Lisinopril', 'Statins', 'Metformin', 'Beta Blockers'], index=0)
exercise_per_week = st.number_input('Exercise per Week (Hours)', min_value=0.0, max_value=10.0, value=3.5, step=0.1)
age = st.number_input('Age', min_value=0, max_value=100, value=45, step=1)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=29.8, step=0.1)
liver_function = st.number_input('Liver Function', min_value=0.0, max_value=10.0, value=6.3, step=0.1)
blood_sugar = st.number_input('Blood Sugar Level', min_value=50.0, max_value=200.0, value=90.5, step=0.1)
smoker = st.selectbox('Smoker', ['No', 'Yes'], index=0)
diabetes = st.selectbox('Diabetes', ['No', 'Yes'], index=0)
diagnosis = st.selectbox('Diagnosis', ['Liver Disease', 'Healthy', 'Kidney Disease', 'Heart Disease', 'Diabetes', 'Hypertension'], index=0)
hospital_stay_days = st.number_input('Hospital Stay Days', min_value=0, max_value=100, value=3, step=1)
hospital_visits = st.number_input('Hospital Visits', min_value=0, max_value=100, value=4, step=1)

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
    mental_health_score, medication_value, exercise_per_week, age, bmi,
    liver_function, blood_sugar, smoker, diabetes, diagnosis_value,
    hospital_stay_days, hospital_visits
]], columns=columns)

# When the 'Predict Treatment Outcome' button is clicked
if st.button('Predict Treatment Outcome'):
    # Make the prediction using the model
    prediction = loaded_model.predict(input_data)

    outcome_mapping = {
        0: 'failure ',
        1: 'improvement ',
        2: 'recovered '
    }

    # Display the result
    st.write(f'Predicted Treatment Outcome: {outcome_mapping[prediction[0]]}')
