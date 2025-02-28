import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import pickle

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

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_processed)

# Save the preprocessor and the model
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
