import streamlit as st
import pandas as pd
from PIL import Image
import io
from utils import preprocess_data, resize_image  # Import your utility functions
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv('shopping_trends.csv')

# Preprocess data and fit KMeans
data_processed, preprocessor = preprocess_data(data)
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_processed)

# Calculate cluster metrics
cluster_info = data.groupby('Cluster').agg({
    'Age': 'mean',
    'Purchase Amount (USD)': 'mean',
    'Previous Purchases': 'mean',
    'Gender': lambda x: (x == 'Male').mean(),  # Percentage Male
    'Customer ID': 'size'  # Cluster size
}).rename(columns={'Age': 'Average Age', 'Gender': 'Percentage Male', 'Customer ID': 'Cluster Size'})

# Streamlit layout
st.title('Customer Segmentation Based on Shopping Trends')

# Data Overview
st.header("Data Overview")
st.dataframe(data)

# Cluster Overview with detailed metrics
st.header("Cluster Overview")
for i in range(4):
    st.subheader(f"Cluster {i}")
    cluster_metrics = cluster_info.loc[i]
    st.write(cluster_metrics)

    # Optional: Most common items or categories for each cluster
    common_items = data[data['Cluster'] == i]['Item Purchased'].value_counts().head(5)
    st.write("Most Commonly Purchased Items:")
    st.write(common_items)

    common_categories = data[data['Cluster'] == i]['Category'].value_counts().head(5)
    st.write("Most Common Categories:")
    st.write(common_categories)

# Sidebar for image upload and resizing
st.sidebar.title("Upload and Resize Images")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    resized_image = resize_image(image, 300)
    st.sidebar.image(resized_image, caption='Resized Image')

# Sidebar for input - existing customer analysis
st.sidebar.title("Customer Profile Analysis")
customer_id = st.sidebar.number_input("Enter Customer ID", min_value=int(data['Customer ID'].min()), max_value=int(data['Customer ID'].max()), step=1)
analyze_button = st.sidebar.button("Submit")

if analyze_button:
    customer_data = data[data['Customer ID'] == customer_id]
    if not customer_data.empty:
        st.sidebar.write("Customer Details:")
        st.sidebar.write(customer_data[['Gender', 'Age', 'Category']])
        cluster_number = customer_data.iloc[0]['Cluster']
        st.sidebar.write(f"This customer belongs to Cluster {cluster_number}.")
    else:
        st.sidebar.write("No customer found with this ID.")
