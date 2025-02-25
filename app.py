import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

# Calculate cluster metrics
cluster_info = data.groupby('Cluster').agg({
    'Age': 'mean',
    'Purchase Amount (USD)': 'mean',
    'Previous Purchases': 'mean',
    'Gender': lambda x: (x == 'Male').mean(),  # Percentage Male
    'Customer ID': 'size'  # Cluster size
}).rename(columns={'Age': 'Average Age', 'Gender': 'Percentage Male', 'Customer ID': 'Cluster Size'})

# Streamlit layout
st.title('Customer Segmentation Based on Shopping Trends')

# Data Overview
st.header("Data Overview")
st.dataframe(data)

st.header("Cluster Overview")
# --- Cluster 0 ---
st.subheader("Cluster 0")
cluster_0_metrics = cluster_info.loc[0]
st.write(cluster_0_metrics)

# รายการสินค้าที่พบบ่อย
common_items_0 = data[data['Cluster'] == 0]['Item Purchased'].value_counts().head(5)
st.write("Most Commonly Purchased Items:")
st.write(common_items_0)

# แสดงพื้นที่สำหรับเพิ่มภาพ
st.image("https://zapaka.com/cdn/shop/products/04033012-Purplefirst.jpg?v=1628925127", caption="Dress", use_container_width=True)

common_categories_0 = data[data['Cluster'] == 0]['Category'].value_counts().head(5)
st.write("Most Common Categories:")
st.write(common_categories_0)

# แสดงพื้นที่สำหรับเพิ่มภาพ
st.image("https://www.thoughtco.com/thmb/ctxxtfGGeK5f_-S3f8J-jbY-Gp8=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/close-up-of-clothes-hanging-in-row-739240657-5a78b11f8e1b6e003715c0ec.jpg", caption="Clothing", use_container_width=True)

# --- Cluster 1 ---
st.subheader("Cluster 1")
cluster_1_metrics = cluster_info.loc[1]
st.write(cluster_1_metrics)

common_items_1 = data[data['Cluster'] == 1]['Item Purchased'].value_counts().head(5)
st.write("Most Commonly Purchased Items:")
st.write(common_items_1)

# แสดงพื้นที่สำหรับเพิ่มภาพ
st.image("https://www.buccellati.com/media/catalog/category/1_High_Jewelry.jpg", caption="Jewelry", use_container_width=True)

common_categories_1 = data[data['Cluster'] == 1]['Category'].value_counts().head(5)
st.write("Most Common Categories:")
st.write(common_categories_1)

# แสดงพื้นที่สำหรับเพิ่มภาพ
st.image("https://www.bigissuenorth.com/wp-content/uploads/2019/05/lyc-campaigns-1600_0_bigissuenorth.jpg", caption="Clothing", use_container_width=True)

# --- Cluster 2 ---
st.subheader("Cluster 2")
cluster_2_metrics = cluster_info.loc[2]
st.write(cluster_2_metrics)

common_items_2 = data[data['Cluster'] == 2]['Item Purchased'].value_counts().head(5)
st.write("Most Commonly Purchased Items:")
st.write(common_items_2)

# แสดงพื้นที่สำหรับเพิ่มภาพ
st.image("https://wildrhinoshoes.com.au/cdn/shop/files/Belts_1.jpg?v=1708316630", caption="Belt", use_container_width=True)

common_categories_2 = data[data['Cluster'] == 2]['Category'].value_counts().head(5)
st.write("Most Common Categories:")
st.write(common_categories_2)

# แสดงพื้นที่สำหรับเพิ่มภาพ
st.image("https://images.theconversation.com/files/580726/original/file-20240308-20-2hk7h6.jpg?ixlib=rb-4.1.0&rect=0%2C167%2C5314%2C2653&q=45&auto=format&w=1356&h=668&fit=crop", caption="Clothing", use_container_width=True)

# --- Cluster 3 ---
st.subheader("Cluster 3")
cluster_3_metrics = cluster_info.loc[3]
st.write(cluster_3_metrics)

common_items_3 = data[data['Cluster'] == 3]['Item Purchased'].value_counts().head(5)
st.write("Most Commonly Purchased Items:")
st.write(common_items_3)

# แสดงพื้นที่สำหรับเพิ่มภาพ
st.image("https://www.mrporter.com/variants/images/1647597332681244/pr/w1000.jpg", caption="Shirt", use_container_width=True)

common_categories_3 = data[data['Cluster'] == 3]['Category'].value_counts().head(5)
st.write("Most Common Categories:")
st.write(common_categories_3)

# แสดงพื้นที่สำหรับเพิ่มภาพ
st.image("https://www.permanentstyle.com/wp-content/uploads/2021/04/hang-up-vintage-london-580x464.jpg", caption="Clothing", use_container_width=True)

# Sidebar for input - existing customer analysis
st.sidebar.title("Customer Profile Analysis")
age_inp = st.sidebar.number_input("Input Age")
purchase_amount_inp = st.sidebar.number_input("Input Purchase Amount(USD)")
previous_purchase_inp = st.sidebar.number_input("Input Previous Purchases")
frequency_purchases_inp = st.sidebar.number_input("Input Frequency of Purchases")
submit_button = st.sidebar.button("Submit")

'''
if submit_button: 
    customer_data = data[data['Customer ID'] == customer_id]
    if not customer_data.empty:
        st.sidebar.write("Customer Details:")
        st.sidebar.write(customer_data[['Gender', 'Age', 'Category']])
        cluster_number = customer_data.iloc[0]['Cluster']
        st.sidebar.write(f"This customer belongs to Cluster {cluster_number}.")
    else:
        st.sidebar.write("No customer found with this ID.")
'''