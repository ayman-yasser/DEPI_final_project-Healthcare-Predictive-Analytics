import mlflow
import pandas as pd
import joblib

data = [
  [
    4.5,  # mental_health_score
    3,    # medication
    3.5,  # exercise_per_week
    45.0, # age
    29.8, # bmi
    6.3,  # liver_function
    90.5, # blood_sugar
    1,    # smoker
    0,    # diabetes
    1,    # diagnosis
    3.0,  # hospital_stay_days
    4.0   # hospital_visits
  ]
]

#logged_model = 'runs:/24ddb137024c4ed6aa653c10096afb22/DecisionTreeClassifier/with-SMOTE'
#loaded_model = mlflow.pyfunc.load_model(logged_model)
loaded_model = joblib.load("model.pkl")

columns = [
    'mental_health_score', 'medication', 'exercise_per_week', 'age', 'bmi',
    'liver_function', 'blood_sugar', 'smoker', 'diabetes', 'diagnosis',
    'hospital_stay_days', 'hospital_visits'
]

input_data = pd.DataFrame(data, columns=columns)


predictions = loaded_model.predict(input_data)
print(predictions)