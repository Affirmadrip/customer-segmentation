import streamlit as st
import pandas as pd
import pickle

# Load the trained model and the preprocessor
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Assuming data is static and pre-loaded
data = pd.read_csv('shopping_trends.csv')
data_processed = preprocessor.transform(data)
data['Cluster'] = kmeans.predict(data_processed)

# Streamlit layout
st.title('Customer Segmentation Based on Shopping Trends')
st.header("Data Overview")
st.dataframe(data)

# Image and caption mappings
image_info = {
    0: [("images/dress.png", "Dress"), ("images/clothing_a.png", "Clothing A")],
    1: [("images/jewelry.png", "Jewelry"), ("images/clothing_b.png", "Clothing B")],
    2: [("images/belt.png", "Belt"), ("images/clothing_c.png", "Clothing C")],
    3: [("images/shirt.png", "Shirt"), ("images/clothing_d.png", "Clothing D")]
}

st.header("Cluster Overview")
for i in range(4):
    st.subheader(f"Cluster {i}")
    cluster_metrics = data.groupby('Cluster').get_group(i).agg({
        'Age': 'mean',
        'Purchase Amount (USD)': 'mean',
        'Previous Purchases': 'mean',
        'Gender': lambda x: (x == 'Male').mean(),
    }).rename({'Age': 'Average Age', 'Gender': 'Percentage Male'})
    st.write(cluster_metrics)

    common_items = data[data['Cluster'] == i]['Item Purchased'].value_counts().head(5)
    st.write("Most Commonly Purchased Items:")
    st.write(common_items.to_frame())
    st.image(image_info[i][0][0], caption=image_info[i][0][1], use_container_width=True)

    common_categories = data[data['Cluster'] == i]['Category'].value_counts().head(5)
    st.write("Most Common Categories:")
    st.write(common_categories.to_frame())
    st.image(image_info[i][1][0], caption=image_info[i][1][1], use_container_width=True)

# Sidebar for customer input
st.sidebar.title("Customer Profile Analysis")
age_inp = st.sidebar.number_input("Input Age")
purchase_amount_inp = st.sidebar.number_input("Input Purchase Amount (USD)")
previous_purchase_inp = st.sidebar.number_input("Input Previous Purchases")
frequency_purchases_inp = st.sidebar.number_input("Input Frequency of Purchases")
submit_button = st.sidebar.button("Submit")

if submit_button:
    user_data = pd.DataFrame({
        'Age': [age_inp],
        'Purchase Amount (USD)': [purchase_amount_inp],
        'Previous Purchases': [previous_purchase_inp],
        'Gender': ['Input Gender Here'],  # Ensure this input matches your data format
        'Category': ['Input Category Here'],
        'Item Purchased': ['Input Item Here']
    })
    user_processed = preprocessor.transform(user_data)
    cluster_prediction = kmeans.predict(user_processed)
    st.sidebar.write(f"Predicted Customer Cluster: {cluster_prediction[0]}")
