import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import pickle

# Load data
data = pd.read_csv('shopping_trends.csv')

# Define the features
numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
categorical_features = ['Gender', 'Category', 'Item Purchased']

# Setup preprocessing steps
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
data_processed = preprocessor.fit_transform(data)

# Train the KMeans model
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_processed)

# Print the parameters of the model
print("KMeans model parameters:", kmeans.get_params())

# Optionally, print preprocessor parameters
print("Preprocessor parameters:", preprocessor.get_params())

# Save the trained model and preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
