import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv('shopping_trends.csv')

# Preprocessing for clustering
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

# Image mappings for each cluster
cluster_images = {
    0: ("images/dress.png", "images/clothing_a.png"),
    1: ("images/jewelry.png", "images/clothing_b.png"),
    2: ("images/belt.png", "images/clothing_c.png"),
    3: ("images/shirt.png", "images/clothing_d.png")
}

# Cluster Overview
for i in range(4):
    st.subheader(f"Cluster {i}")
    cluster_metrics = cluster_info.loc[i]
    st.write(cluster_metrics)

    # Most Commonly Purchased Items
    common_items = data[data['Cluster'] == i]['Item Purchased'].value_counts().head(5)
    st.write("Most Commonly Purchased Items:")
    st.write(common_items.to_frame())
    # Display item image
    st.image(cluster_images[i][0], caption="Most Commonly Purchased Item", use_container_width=True)

    # Most Common Categories
    common_categories = data[data['Cluster'] == i]['Category'].value_counts().head(5)
    st.write("Most Common Categories:")
    st.write(common_categories.to_frame())
    # Display category image
    st.image(cluster_images[i][1], caption="Most Common Category", use_container_width=True)

# Sidebar for input - existing customer analysis
st.sidebar.title("Customer Profile Analysis")
age_inp = st.sidebar.number_input("Input Age")
purchase_amount_inp = st.sidebar.number_input("Input Purchase Amount(USD)")
previous_purchase_inp = st.sidebar.number_input("Input Previous Purchases")
frequency_purchases_inp = st.sidebar.number_input("Input Frequency of Purchases")
submit_button = st.sidebar.button("Submit")
