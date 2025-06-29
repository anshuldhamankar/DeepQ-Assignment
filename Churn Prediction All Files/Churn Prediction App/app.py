import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

#load model and feature columns
model = joblib.load('churn_model_lgbm.pkl')
feature_columns = joblib.load("feature_columns.pkl")
categorical_columns = joblib.load("categorical_columns.pkl")  # saved during training

#get top 15 important features
importances = model.feature_importances_
top_features = pd.Series(importances, index=feature_columns).sort_values(ascending=False).head(15).index.tolist()

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("🔍 Customer Churn Prediction")
st.markdown("Enter customer details for top features to predict churn probability.")

#Build input dictionary
user_input = {}

#Create columns for UI input
num_cols = 3
cols = st.columns(num_cols)
chunks = [top_features[i::num_cols] for i in range(num_cols)]

#input form for top features
for i in range(num_cols):
    with cols[i]:
        for col in chunks[i]:
            user_input[col] = st.text_input(f"{col}", value="")

#prediction section
if st.button("Predict Churn Probability"):
    try:
        #create a full input row with all expected features
        full_input = {col: 0 for col in feature_columns}
        for k, v in user_input.items():
            try:
                full_input[k] = float(v)
            except:
                pass  # leave default 0

        input_df = pd.DataFrame([full_input])

        # cast categorical columns to 'category'
        for col in categorical_columns:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype("category")

        # predict churn probability
        prob = model.predict_proba(input_df)[0][1]
        st.success(f"🧠 Predicted Churn Probability: **{prob:.2%}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
