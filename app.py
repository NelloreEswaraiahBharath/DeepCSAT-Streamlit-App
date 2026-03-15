import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("csat_model.pkl")

st.title("DeepCSAT - Customer Satisfaction Prediction")

st.write("Enter customer support details to predict CSAT score")

# Example inputs
channel_name = st.number_input("Channel Name", min_value=0)
category = st.number_input("Product Category", min_value=0)
sub_category = st.number_input("Sub Category", min_value=0)
customer_city = st.number_input("Customer City Code", min_value=0)
order_hour = st.slider("Order Hour", 0, 23)
order_day = st.slider("Order Day", 1, 31)
order_month = st.slider("Order Month", 1, 12)

# Convert input to dataframe
input_data = pd.DataFrame({
    'channel_name':[channel_name],
    'Product_category':[category],
    'Sub_category':[sub_category],
    'Customer_City':[customer_city],
    'order_hour':[order_hour],
    'order_day':[order_day],
    'order_month':[order_month]
})

if st.button("Predict CSAT Score"):
    
    prediction = model.predict(input_data)
    
    st.success(f"Predicted CSAT Score: {prediction[0]}")