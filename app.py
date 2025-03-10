import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'  # Start on the home page

# Function to navigate to home
def go_home():
    st.session_state.page = 'home'


# Load the trained model and preprocessor
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('Model.pkl', 'rb') as f:
    model = pickle.load(f)

def spending_score(Purchase, Previous, Frequency):
    freq_per_year = {
        'Bi-Weekly': 104,
        'Weekly': 52,
        'Fortnightly': 26,
        'Monthly': 12,
        'Every 3 Months': 4,
        'Quarterly': 3,
        'Annually': 1
    }
    w1 = 1/3
    w2 = 1/3
    w3 = 1/3

    # Instead of applying the lambda, just directly map the Frequency to its corresponding value
    FreqPerYear = freq_per_year.get(Frequency, 0)  # Default to 0 if the Frequency isn't found

    # Calculate the spending score
    return w1 * (Purchase / 100) + w2 * (Previous / 50) + w3 * (FreqPerYear / 104)


# Load the data
data = pd.read_csv('shopping_trends_cluster.csv')

# Ensure "Cluster" column exists
if 'Cluster' not in data.columns:
    st.error("Error: 'Cluster' column is missing. Run trained_model.py first.")
    st.stop()


# Calculate cluster metrics
cluster_info = data.groupby('Cluster').agg({
    'Age': 'mean',
    'Purchase Amount (USD)': 'mean',
    'Previous Purchases': 'mean',
    # 'Gender': lambda x: (x == 'Male').mean(),  # Percentage of Males
    'Customer ID': 'size'  # Cluster size
}).rename(columns={'Age': 'Average Age', 'Customer ID': 'Cluster Size'})

# Streamlit layout for the home page
if st.session_state.page == 'home':
    st.title('Customer Segmentation Based on Shopping Trends')
    st.header("Data Overview")
    st.dataframe(data)

    # Image and caption mappings (Updated)
    image_info = {
        0: [("images/shirt.png", "Shirt"), ("images/pants-men.png", "Pants"), ("images/hoodie.png", "Hoodie")],
        1: [("images/jacket.png", "Jacket"), ("images/handbag-men.png", "Handbag"), ("images/sandals.png", "Sandals")],
        2: [("images/jewelry.png", "Jewelry"), ("images/sweater.png", "Sweater"), ("images/coat.png", "Coat")],
    }

    # Cluster Overview
    st.header("Cluster Overview")
    for i in range(3):
        st.subheader(f"Cluster {i}")
        st.dataframe(cluster_info.loc[i].to_frame().T)  # Display as a table

        # Display 3 images per cluster in columns
        cols = st.columns(3)
        for idx, (img_path, caption) in enumerate(image_info[i]):
            with cols[idx]:
                st.image(img_path, caption=caption, use_container_width=True)

    # Sidebar for Customer Input
    st.sidebar.title("Customer Profile Analysis")
    age_inp = st.sidebar.number_input("Input Age", min_value=0, step=1, format="%d")
    purchase_amount_inp = st.sidebar.number_input("Input Purchase Amount (USD)", min_value=0, step=1, format="%d")  # Changed to integer
    previous_purchase_inp = st.sidebar.number_input("Input Previous Purchases", min_value=0, step=1, format="%d")
    frequency_purchases_inp = st.sidebar.selectbox("Select Frequency of Purchases", 
                         options=['Bi-Weekly', 'Weekly', 'Fortnightly', 'Monthly', 'Every 3 Months', 'Quarterly', 'Annually'])
    predict_button = st.sidebar.button("Predict")

    # Perform Prediction
    if predict_button:
        SpendingScore_data = spending_score(purchase_amount_inp, previous_purchase_inp, frequency_purchases_inp)
        input_data = pd.DataFrame([[age_inp,SpendingScore_data]])
        processed_data = scaler.transform(input_data)
        cluster_prediction = model.predict(processed_data)[0]

        # Show predicted cluster result
        st.sidebar.subheader(f"Predicted Cluster: {cluster_prediction}")

        # Navigate to page of cluster prediction
        if cluster_prediction == 0:
            st.session_state.page = 'page1'
            st.rerun()  # Refresh the app 
        elif cluster_prediction == 1:
            st.session_state.page = 'page2'
            st.rerun()  # Refresh the app 
        elif cluster_prediction == 2:
            st.session_state.page = 'page3'
            st.rerun()  # Refresh the app 

# Page 1
elif st.session_state.page == 'page1':
    st.title("Cluster 0")
    
    # Use st.components.v1.html to render the iframe
    st.components.v1.html(
        """
        <iframe
            src="https://cluster-0.streamlit.app?embed=true"
            style="height: 450px; width: 100%;"
            frameborder="0">
        </iframe>
        """,
        height=450,  
        width=700,   
    )

    # Button to go back to Home
    if st.button("Back to Home"):
        go_home()

# Page 2
elif st.session_state.page == 'page2':
    st.title("Cluster 1")
    
    # Use st.components.v1.html to render the iframe
    st.components.v1.html(
        """
        <iframe
            src="https://cluster-1.streamlit.app?embed=true"
            style="height: 450px; width: 100%;"
            frameborder="0">
        </iframe>
        """,
        height=450,  
        width=700,   
    )
    
    # Button to go back to Home
    if st.button("Back to Home"):
        go_home()

# Page 3
elif st.session_state.page == 'page3':
    st.title("Cluster 2")
    
    # Use st.components.v1.html to render the iframe
    st.components.v1.html(
        """
        <iframe
            src="https://cluster-2.streamlit.app?embed=true"
            style="height: 450px; width: 100%;"
            frameborder="0">
        </iframe>
        """,
        height=450,  
        width=700,   
    )

    # Button to go back to Home
    if st.button("Back to Home"):
        go_home()