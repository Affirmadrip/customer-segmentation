import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import numpy as np

# Load data
data = pd.read_csv('shopping_trends.csv')

# Data preprocessing for clustering
numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
categorical_features = ['Gender', 'Category']

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
    'Gender': lambda x: (x == 'Male').mean(),
    'Customer ID': 'size'
}).rename(columns={'Age': 'Average Age', 'Gender': 'Percentage Male', 'Customer ID': 'Cluster Size'})

# Streamlit layout
st.title('Customer Segmentation Analysis')

# Data Overview
st.header("Data Overview")
st.dataframe(data)  # Allows user to scroll through the data

# Cluster Overview
st.header("Cluster Overview")
for i in range(4):
    st.subheader(f"Cluster {i}")
    cluster_subset = cluster_info.loc[i]
    st.write(f"Average Age: {cluster_subset['Average Age']:.2f}")
    st.write(f"Percentage Male: {cluster_subset['Percentage Male'] * 100:.2f}%")
    st.write(f"Cluster Size: {cluster_subset['Cluster Size']}")

    # Gender distribution within the cluster
    gender_count = data[data['Cluster'] == i]['Gender'].value_counts()
    fig = sns.barplot(x=gender_count.index, y=gender_count.values)
    st.pyplot(fig)
