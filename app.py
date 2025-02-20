import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('shopping_trends.csv')

# Preprocessing
numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
categorical_features = ['Gender', 'Category']
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

data_preprocessed = preprocessor.fit_transform(data)
features = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
data_preprocessed_df = pd.DataFrame(data_preprocessed, columns=features)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_preprocessed_df)

# Streamlit Layout
st.title('Customer Segmentation Analysis')
st.sidebar.title("Customer Profile Analysis")

# Sidebar for input
customer_id = st.sidebar.number_input("Customer ID", min_value=int(data['Customer ID'].min()), max_value=int(data['Customer ID'].max()), value=int(data['Customer ID'].min()))
gender = st.sidebar.radio("Gender", ('Male', 'Female'))
age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
analyze_button = st.sidebar.button("Analyze Customer")

# Cluster Overview
st.header("Cluster Overview")
if analyze_button:
    customer_data = {'Customer ID': [customer_id], 'Gender': [gender], 'Age': [age]}
    customer_df = pd.DataFrame(customer_data)
    # Dummy data processing - in real use, integrate with model predictions
    # Display cluster info and recommended products (Placeholder)
    st.write("Customer belongs to Cluster:", kmeans.predict(preprocessor.transform(customer_df))[0])

# Data Overview
st.header("Data Overview")
st.write(data.describe())

# Visualization of Cluster Distribution
st.header("Cluster Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Cluster', data=data, palette='viridis', ax=ax)
st.pyplot(fig)

# Save this script as `app.py` and ensure your dataset `shopping_trends.csv` is in the same directory.
