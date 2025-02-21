import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
#pear
#Gim
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

# Sidebar for input - existing customer analysis
st.sidebar.title("Customer Profile Analysis")
customer_id = st.sidebar.number_input("Enter Customer ID", min_value=int(data['Customer ID'].min()), max_value=int(data['Customer ID'].max()), step=1)
analyze_button = st.sidebar.button("Analyze Customer")

if analyze_button:
    customer_data = data[data['Customer ID'] == customer_id]
    if not customer_data.empty:
        st.sidebar.write("Customer Details:")
        st.sidebar.write(customer_data[['Gender', 'Age', 'Category']])
        cluster_number = customer_data.iloc[0]['Cluster']
        st.sidebar.write(f"This customer belongs to Cluster {cluster_number}.")
    else:
        st.sidebar.write("No customer found with this ID.")
