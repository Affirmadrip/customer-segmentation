import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# Page configuration
st.set_page_config(page_title="Well Shop", layout="wide")

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

# Initialize session state
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

def make_prediction(age, purchase_amount, review_rating, previous_purchases):
    input_data = pd.DataFrame([[age, purchase_amount, review_rating, previous_purchases]],
                              columns=numerical_features)
    input_transformed = preprocessor.transform(input_data)
    cluster_pred = kmeans.predict(input_transformed)
    return cluster_pred[0]

if not st.session_state.predicted:
    # Sidebar for customer input
    st.sidebar.title("Customer Profile Analysis")
    age_inp = st.sidebar.number_input("Input Age")
    purchase_amount_inp = st.sidebar.number_input("Input Purchase Amount(USD)")
    previous_purchase_inp = st.sidebar.number_input("Input Previous Purchases")
    frequency_purchases_inp = st.sidebar.number_input("Input Frequency of Purchases")
    submit_button = st.sidebar.button("Submit")

    if submit_button:
        # Make prediction
        predicted_cluster = make_prediction(age_inp, purchase_amount_inp, previous_purchase_inp, frequency_purchases_inp)
        st.session_state.predicted = True
        st.session_state.cluster = predicted_cluster

    # Cluster Information Display
    st.title('Customer Segmentation Based on Shopping Trends')
    st.header("Data Overview")
    st.dataframe(data)

    st.header("Cluster Overview")
    for i in range(4):
        st.subheader(f"Cluster {i}")
        cluster_metrics = cluster_info.loc[i]
        st.write(cluster_metrics)

        common_items = data[data['Cluster'] == i]['Item Purchased'].value_counts().head(5)
        st.write("Most Commonly Purchased Items:")
        st.write(common_items.to_frame())

        common_categories = data[data['Cluster'] == i]['Category'].value_counts().head(5)
        st.write("Most Common Categories:")
        st.write(common_categories.to_frame())

else:
    # Promotional page after prediction
    promo_image = "your_encoded_image_string_here"
    st.markdown(f"""
        <div class='banner'>
            <img src='{promo_image}' alt='Promotion'>
            <div class='banner-text'>🔥 Special Promotion - Limited Time! 🔥</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='title'>Well Shop</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Our Products</div>", unsafe_allow_html=True)
    
    # Simulated product display (replace with actual product details and links)
    # Display products in a grid format
    # For example, you can use st.columns or HTML/CSS in markdown to arrange the layout

    # Reset button to allow a new prediction
    if st.button("New Prediction"):
        st.session_state.predicted = False
