import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model and preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Define numerical features
numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']

# Load dataset for displaying cluster information
data = pd.read_csv('shopping_trends.csv')

# Apply clustering to data
data_processed = preprocessor.transform(data)
data['Cluster'] = kmeans.predict(data_processed)

# Calculate cluster metrics
cluster_info = data.groupby('Cluster').agg({
    'Age': 'mean',
    'Purchase Amount (USD)': 'mean',
    'Previous Purchases': 'mean',
    'Gender': lambda x: (x == 'Male').mean(),  # Percentage Male
    'Customer ID': 'size'  # Cluster size
}).rename(columns={'Age': 'Average Age', 'Gender': 'Percentage Male', 'Customer ID': 'Cluster Size'})

# Streamlit UI Layout
st.title('Customer Segmentation Based on Shopping Trends')

st.header("Data Overview")
st.dataframe(data)

# Image and caption mappings
image_info = {
    0: [("images/dress.png", "Dress"), ("images/clothing_a.png", "Clothing A")],
    1: [("images/jewelry.png", "Jewelry"), ("images/clothing_b.png", "Clothing B")],
    2: [("images/belt.png", "Belt"), ("images/clothing_c.png", "Clothing C")],
    3: [("images/shirt.png", "Shirt"), ("images/clothing_d.png", "Clothing D")]
}

st.header("Cluster Overview")
for i in range(4):
    st.subheader(f"Cluster {i}")
    cluster_metrics = cluster_info.loc[i]
    st.dataframe(cluster_metrics.to_frame().T)  # **Fixed table format**

    # Most Commonly Purchased Items
    common_items = data[data['Cluster'] == i]['Item Purchased'].value_counts().head(5)
    st.write("Most Commonly Purchased Items:")
    st.dataframe(common_items.to_frame())

    st.image(image_info[i][0][0], caption=image_info[i][0][1], use_container_width=True)

    # Most Common Categories
    common_categories = data[data['Cluster'] == i]['Category'].value_counts().head(5)
    st.write("Most Common Categories:")
    st.dataframe(common_categories.to_frame())

    st.image(image_info[i][1][0], caption=image_info[i][1][1], use_container_width=True)

# Sidebar Input
st.sidebar.title("Customer Profile Prediction")

age_inp = st.sidebar.number_input("Input Age")
purchase_amount_inp = st.sidebar.number_input("Input Purchase Amount (USD)")
previous_purchase_inp = st.sidebar.number_input("Input Previous Purchases")
frequency_purchases_inp = st.sidebar.number_input("Input Frequency of Purchases")  # **Restored input**
submit_button = st.sidebar.button("Predict")  # **Renamed from "Analyze"**

if submit_button:
    try:
        # Prepare input data for prediction
        input_data = pd.DataFrame([[age_inp, purchase_amount_inp, previous_purchase_inp, frequency_purchases_inp]],
                                  columns=['Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Review Rating'])

        input_processed = preprocessor.transform(input_data)
        cluster_prediction = kmeans.predict(input_processed)[0]

        st.sidebar.success(f"The predicted cluster is: {cluster_prediction}")
    except Exception as e:
        st.sidebar.error(f"Prediction failed: {e}")
