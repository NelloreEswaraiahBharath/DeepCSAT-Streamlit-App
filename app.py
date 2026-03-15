import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("DeepCSAT - Customer Satisfaction Prediction")

# Load model
model = joblib.load("csat_model.pkl")

st.write("Enter customer support details")

channel_name = st.number_input("Channel Name", 0, 10, 1)
product_category = st.number_input("Product Category", 0, 10, 2)
sub_category = st.number_input("Sub Category", 0, 20, 3)
customer_city = st.number_input("Customer City Code", 0, 100, 10)

order_hour = st.slider("Order Hour", 0, 23, 12)
order_day = st.slider("Order Day", 1, 31, 15)
order_month = st.slider("Order Month", 1, 12, 6)

if st.button("Predict CSAT Score"):

    # Create base input
    input_data = pd.DataFrame({
        'channel_name':[channel_name],
        'Product_category':[product_category],
        'Sub_category':[sub_category],
        'Customer_City':[customer_city],
        'order_hour':[order_hour],
        'order_day':[order_day],
        'order_month':[order_month]
    })

    # Align with model features
    expected_features = model.feature_names_in_

    final_input = pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)

    for col in input_data.columns:
        if col in final_input.columns:
            final_input[col] = input_data[col]

    prediction = model.predict(final_input)

    st.success(f"Predicted CSAT Score: {prediction[0]}")