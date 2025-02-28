import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import pickle

# Load data
data = pd.read_csv('shopping_trends.csv')

# Preprocessing for clustering
numerical_features = data[['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']].copy()

preprocessor = StandardScaler()

# Fit the preprocessor on the data
data_processed = preprocessor.fit_transform(numerical_features)

# Train KMeans model
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_processed)



print(preprocessor.get_params())

