import pickle
import numpy as np
import streamlit as st

import os
model_path = os.path.join(os.path.dirname(__file__), 'diabetes_model.sav')

# function for prediction
def diabetes_prediction(input_data):

    # convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # reshape for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # prediction
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "The person is **not diabetic**"
    else:
        return "The person is **diabetic**"


# Streamlit UI
st.title("🔮 Diabetes Prediction App")

Pregnancies = st.number_input("Pregnancies", min_value=0)
Glucose = st.number_input("Glucose Level", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin Level", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                  Insulin, BMI, DiabetesPedigree, Age]

    result = diabetes_prediction(user_input)
    st.success(result)
