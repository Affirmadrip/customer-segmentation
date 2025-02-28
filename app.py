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

# Define function to predict customer cluster
def predict_customer_cluster(age, purchase_amount, previous_purchases):
    # Assume review rating is neutral if not provided
    review_rating = 3  
    # Create a DataFrame for the new customer data
    new_customer = pd.DataFrame({
        'Age': [age],
        'Purchase Amount (USD)': [purchase_amount],
        'Review Rating': [review_rating],
        'Previous Purchases': [previous_purchases]
    })
    # Transform the new customer data using the preprocessor
    new_customer_transformed = preprocessor.transform(new_customer)
    # Predict the cluster
    cluster = kmeans.predict(new_customer_transformed)
    return cluster[0]

# Streamlit app layout
st.set_page_config(page_title="Customer Clustering & Promotions", layout="wide")

# Sidebar for customer input
st.sidebar.title("Customer Profile Analysis")
age = st.sidebar.number_input("Input Age", min_value=18, max_value=100, step=1)
purchase_amount = st.sidebar.number_input("Input Purchase Amount (USD)", min_value=1, max_value=10000, step=1)
previous_purchases = st.sidebar.number_input("Input Previous Purchases", min_value=0, max_value=100, step=1)
submit_button = st.sidebar.button("Predict Cluster")

if submit_button:
    # Predict the cluster for the given inputs
    predicted_cluster = predict_customer_cluster(age, purchase_amount, previous_purchases)
    st.session_state['predicted_cluster'] = predicted_cluster
    st.session_state['show_promotions'] = True  # Trigger to switch to the promotional page

# Check whether to show the promotional page or the main overview
if 'show_promotions' in st.session_state and st.session_state['show_promotions']:
    # Show promotional page
    st.markdown("## Special Promotions Based on Your Shopping Preferences")
    st.markdown(f"### You are in Cluster {st.session_state['predicted_cluster']}")
    # Include your promotional page setup here
else:
    # Show the main clustering overview
    st.title('Customer Segmentation Based on Shopping Trends')
    st.header("Cluster Overview")
    for i in range(4):
        st.subheader(f"Cluster {i}")
        cluster_metrics = cluster_info.loc[i]
        st.write(cluster_metrics)

        # Most Commonly Purchased Items
        common_items = data[data['Cluster'] == i]['Item Purchased'].value_counts().head(5)
        st.write("Most Commonly Purchased Items:")
        st.write(common_items.to_frame())

        # Most Common Categories
        common_categories = data[data['Cluster'] == i]['Category'].value_counts().head(5)
        st.write("Most Common Categories:")
        st.write(common_categories.to_frame())

# Optional: Button to return to the main page from promotions
if 'show_promotions' in st.session_state and st.session_state['show_promotions']:
    if st.button("Return to Main Overview"):
        st.session_state['show_promotions'] = False
        st.experimental_rerun()
