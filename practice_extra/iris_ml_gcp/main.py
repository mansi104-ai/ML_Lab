import streamlit as st
import pickle
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'savedmodel.sav')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit Interface
st.title("Iris Flower Prediction")

# Create a form for input fields
with st.form("iris_form"):
    sepal_length = st.number_input("Sepal Length", min_value=0.0, step=0.1)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, step=0.1)
    petal_length = st.number_input("Petal Length", min_value=0.0, step=0.1)
    petal_width = st.number_input("Petal Width", min_value=0.0, step=0.1)

    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# Process the prediction when the form is submitted
if submit_button:
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"The predicted class is: {result[0]}")
