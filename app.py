import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# Load the preprocessor and model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Streamlit layout
st.title('Customer Segmentation Based on Shopping Trends')

# Load data (for display purposes)
data = pd.read_csv('shopping_trends.csv')
data_processed = preprocessor.transform(data)
data['Cluster'] = kmeans.predict(data_processed)

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
    cluster_info = data[data['Cluster'] == i].agg({
        'Age': 'mean',
        'Purchase Amount (USD)': 'mean',
        'Previous Purchases': 'mean',
        'Gender': lambda x: (x == 'Male').mean()  # Percentage Male
    }).rename({
        'Age': 'Average Age',
        'Gender': 'Percentage Male'
    }).to_dict()

    st.write(cluster_info)

    # Display images with captions for each cluster
    for img, caption in image_info[i]:
        st.image(img, caption=caption, use_container_width=True)

# Sidebar for customer input (Example of input handling)
st.sidebar.title("Customer Profile Analysis")
age_inp = st.sidebar.number_input("Input Age")
purchase_amount_inp = st.sidebar.number_input("Input Purchase Amount (USD)")
previous_purchase_inp = st.sidebar.number_input("Input Previous Purchases")
submit_button = st.sidebar.button("Analyze")

if submit_button:
    # Example prediction logic (simplified for this context)
    # In practice, input features would need to match the model's feature set
    input_data = pd.DataFrame([[age_inp, purchase_amount_inp, 0, previous_purchase_inp]],
                              columns=numerical_features)
    input_processed = preprocessor.transform(input_data)
    cluster = kmeans.predict(input_processed)
    st.sidebar.write(f"Predicted Customer Cluster: {cluster[0]}")
