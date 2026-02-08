import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="SmartCart – Purchase Prediction",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 SmartCart – Customer Purchase Prediction")
st.write(
    "Predict whether a customer is likely to complete a purchase based on their browsing behavior."
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("smartcart_customers.csv")
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Target & Features
# --------------------------------------------------
# ⚠️ CHANGE THIS if your column name is different
TARGET_COLUMN = "Purchase"  # example: 0 = No, 1 = Yes

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# --------------------------------------------------
# Train Model
# --------------------------------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return pipeline, acc

model, accuracy = train_model(X, y)

st.success(f"✅ Model trained successfully | Accuracy: **{accuracy:.2f}**")

# --------------------------------------------------
# User Input Section
# --------------------------------------------------
st.subheader("🧾 Enter Customer Session Details")

user_input = {}

for col in X.columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())

    user_input[col] = st.slider(
        label=col,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

input_df = pd.DataFrame([user_input])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.subheader("🔮 Prediction Result")

if st.button("Predict Purchase"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("🟢 Customer is **LIKELY** to make a purchase")
    else:
        st.error("🔴 Customer is **UNLIKELY** to make a purchase")

    st.metric(
        label="Purchase Probability",
        value=f"{probability * 100:.2f}%"
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | SmartCart Project")
