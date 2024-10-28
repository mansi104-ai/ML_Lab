# streamlit_app.py
import streamlit as st
import pickle
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'savedmodel.sav')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit Interface
st.title("Iris Flower Prediction")
sepal_length = st.number_input("Sepal Length", value=5.0)
sepal_width = st.number_input("Sepal Width", value=3.5)
petal_length = st.number_input("Petal Length", value=1.4)
petal_width = st.number_input("Petal Width", value=0.2)

if st.button("Predict"):
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"The predicted class is: {result[0]}")
