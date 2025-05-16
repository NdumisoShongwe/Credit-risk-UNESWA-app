# Paste code below
import streamlit as st

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("Credit Risk Prediction App")
st.write("👋 Hello! This is a basic Streamlit app template.")
st.write("Once you upload your model and data, predictions and explanations will appear here.")

# Add a sample input field just to make the app interactive
name = st.text_input("Enter your name:")
if name:
    st.success(f"Welcome, {name}!")
