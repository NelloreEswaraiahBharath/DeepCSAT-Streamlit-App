import streamlit as st
import pandas as pd
import joblib

st.title("DeepCSAT - Customer Satisfaction Prediction")

st.write("Enter customer support details to predict CSAT score")

# Load model
model = joblib.load("csat_model.pkl")

# User Inputs
channel_name = st.number_input("Channel Name", min_value=0, value=1)
product_category = st.number_input("Product Category", min_value=0, value=2)
sub_category = st.number_input("Sub Category", min_value=0, value=3)
customer_city = st.number_input("Customer City Code", min_value=0, value=10)

order_hour = st.slider("Order Hour", 0, 23, 12)
order_day = st.slider("Order Day", 1, 31, 15)
order_month = st.slider("Order Month", 1, 12, 6)

if st.button("Predict CSAT Score"):

    # Create dataframe with expected columns
    input_data = pd.DataFrame({
        'channel_name':[channel_name],
        'Product_category':[product_category],
        'Sub_category':[sub_category],
        'Customer_City':[customer_city],
        'order_hour':[order_hour],
        'order_day':[order_day],
        'order_month':[order_month]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted CSAT Score: {prediction[0]}")