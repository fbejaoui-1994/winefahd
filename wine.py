import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title
st.title("🍷 Wine Quality Prediction")

# Load the dataset (you need the CSV file in the same folder or specify path)
@st.cache_data
def load_data():
    data = pd.read_csv("wine_quality_classification.csv")
    return data

data = load_data()

# Prepare data
quality_map = {'low': 0, 'medium': 1, 'high': 2}
data['quality_label'] = data['quality_label'].map(quality_map)

X = data.drop(columns=['quality_label'])
y = data['quality_label']

# Train model once on startup
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

st.sidebar.header("Input Wine Features")

def user_input_features():
    features = {}
    for col in X.columns:
        # Provide numeric input with reasonable defaults and ranges
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())
        features[col] = st.sidebar.number_input(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)

# Map back prediction to label
label_map = {0: 'low', 1: 'medium', 2: 'high'}
pred_label = label_map.get(prediction, "Unknown")

st.subheader("Prediction")
st.write(f"Predicted Wine Quality: **{pred_label.upper()}**")

st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=['Low', 'Medium', 'High'])
st.write(proba_df.T)

