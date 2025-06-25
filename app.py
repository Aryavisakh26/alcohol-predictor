import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Alcohol Consumption Predictor 🍺")

beer = st.number_input("beer servings", min_value=0)
spirit = st.number_input("spirit servings", min_value=0)
wine = st.number_input("wine servings", min_value=0)

continent_options = [
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "Oceania",
    "South America",
]
continent = st.selectbox("continent", continent_options)
continent_mapping = {name: i for i, name in enumerate(continent_options)}
continent_encoded = continent_mapping[continent]


if st.button("Predict Total Alcohol Consumption"):
    features = np.array([[beer, spirit, wine, continent_encoded]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated Total Alcohol Consumption: {prediction:.2f} litres")
