import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv('shopping_trends.csv')

# Define feature columns
numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
categorical_features = ['Gender', 'Category', 'Item Purchased']

# Define preprocessors
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the data
data_processed = preprocessor.fit_transform(data)

# Train K-Means model
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_processed)

# Save the preprocessor and model as pickle files
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
