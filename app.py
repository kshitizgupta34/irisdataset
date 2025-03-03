import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="wide")

st.title("Iris Flower Classification")

model_iris = pickle.load(open('model_iris.pkl','rb'))

# User input fields
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

user_input={'sepal length (cm)':sepal_length,
    'sepal width (cm)': sepal_width,
    'petal length (cm)':petal_length,
   'petal width (cm)':petal_width
   }

user_input_df=pd.DataFrame(user_input,index=[0])

if st.button("Predict"):
    prediction = model_iris.predict(user_input_df)
    #species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    predicted_species = prediction[0]

    # Display result
    st.success(f"The predicted species is: **{predicted_species}**")