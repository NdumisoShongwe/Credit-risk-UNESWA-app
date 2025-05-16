# Paste code below
import streamlit as st
streamlit
pandas
scikit-learn
shap
lime
matplotlib

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("Credit Risk Prediction App")
st.write("👋 Hello! This is a basic Streamlit app template.")
st.write("Once you upload your model and data, predictions and explanations will appear here.")

# Add a sample input field just to make the app interactive
name = st.text_input("Enter your name:")
if name:
    st.success(f"Welcome, {name}!")

import streamlit as st
import pandas as pd
import shap
import lime
import lime.lime_tabular
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# App title and introduction
st.title("Credit Risk Prediction App")
st.markdown("👋 Hello! This is a basic Streamlit app template.")
st.markdown("Once you upload your model and data, predictions and explanations will appear here.")

# Input name
name = st.text_input("Enter your name:")
if name:
    st.success(f"Welcome, {name}!")

# Load model and data
@st.cache_resource
def load_model_data():
    model = pickle.load(open("mlp_model (1).pkl", "rb"))
    X_train = pd.read_csv("X_train (1).csv")
    return model, X_train

model, X_train = load_model_data()

# Explainability Section
st.header("🔍 Model Explainability")

# Select index
index = st.slider("Select a data point to explain", 0, len(X_train) - 1, 0)
sample = X_train.iloc[[index]]
st.write("### Selected Input")
st.write(sample)

# --- SHAP Explanation ---
st.subheader("SHAP Explanation")

# Use KernelExplainer for compatibility with MLP
explainer_shap = shap.Explainer(model.predict, X_train, algorithm="permutation")
shap_values = explainer_shap(sample)

# Plot SHAP explanation
fig_shap = shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(bbox_inches="tight")

# --- LIME Explanation ---
st.subheader("LIME Explanation")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Not Default", "Default"],
    mode="classification"
)

lime_exp = lime_explainer.explain_instance(
    sample.values[0], model.predict_proba, num_features=10
)

st.components.v1.html(lime_exp.as_html(), height=800)
