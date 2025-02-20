import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Customer Segmentation using K-Means Clustering')

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Reading the uploaded file directly
    data = pd.read_csv(uploaded_file)
    st.write("Data Successfully Uploaded")
    st.write(data.head())

    # Data preprocessing
    numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    categorical_features = ['Gender', 'Category']

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Applying transformations
    data_preprocessed = preprocessor.fit_transform(data)
    # To create a DataFrame from the transformed data, we need feature names:
    categorical_encoded_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features = numerical_features + list(categorical_encoded_features)
    data_preprocessed_df = pd.DataFrame(data_preprocessed, columns=all_features)

    # K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(data_preprocessed_df)
    data['Cluster'] = cluster_labels

    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='Purchase Amount (USD)', hue='Cluster', data=data, palette='viridis', ax=ax)
    st.pyplot(fig)
else:
    st.info('Awaiting for CSV file to be uploaded.')
