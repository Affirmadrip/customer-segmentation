import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from image_utils import display_image  # Ensure this module is correctly defined to manage image display

# Load and preprocess data
data = pd.read_csv('shopping_trends.csv')
numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
categorical_features = ['Gender', 'Category', 'Item Purchased']
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

data_processed = preprocessor.fit_transform(data)
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_processed)

# Cluster metrics
cluster_info = data.groupby('Cluster').agg({
    'Age': 'mean',
    'Purchase Amount (USD)': 'mean',
    'Previous Purchases': 'mean',
    'Gender': lambda x: (x == 'Male').mean(),
    'Customer ID': 'size'
}).rename(columns={'Age': 'Average Age', 'Gender': 'Percentage Male', 'Customer ID': 'Cluster Size'})

# Streamlit layout
st.title('Customer Segmentation Based on Shopping Trends')

# Sidebar for customer analysis
st.sidebar.title("Customer Profile Analysis")
age_inp = st.sidebar.number_input("Input Age")
purchase_amount_inp = st.sidebar.number_input("Input Purchase Amount(USD)")
previous_purchase_inp = st.sidebar.number_input("Input Previous Purchases")
frequency_purchases_inp = st.sidebar.number_input("Input Frequency of purchases")
submit_button = st.sidebar.button("Submit")

if submit_button:
    # Here you could add code to process the inputs, for example:
    st.sidebar.write("Received inputs:")
    st.sidebar.write(f"Age: {age_inp}, Purchase Amount: {purchase_amount_inp}")
    st.sidebar.write(f"Previous Purchases: {previous_purchase_inp}, Frequency of Purchases: {frequency_purchases_inp}")
    # Add any processing or display logic you might need

# Cluster Overview
st.header("Cluster Overview")
for i in range(4):
    st.subheader(f"Cluster {i}")
    cluster_metrics = cluster_info.loc[i]
    st.write(cluster_metrics)
    common_items = data[data['Cluster'] == i]['Item Purchased'].value_counts().head(5)
    st.write("Most Commonly Purchased Items:")
    st.write(common_items)
    for item in common_items.index[:1]:  # Display the top item's image
        display_image(st, item)

# Note: Implement the display_image function or ensure it can handle the requests
