# !pip install scikit-learn-extra
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids  # Import KMedoids from sklearn_extra

df = pd.read_csv('shopping_trends.csv')

freq_per_year = {
    'Bi-Weekly': 104,
    'Weekly': 52,
    'Fortnightly': 26,
    'Monthly': 12,
    'Every 3 Months': 4,
    'Quarterly': 3,
    'Annually': 1
}

df['Frequency per Year'] = df['Frequency of Purchases'].apply(lambda x: freq_per_year.get(x))
w1 = 1/3
w2 = 1/3
w3 = 1/3
df2 = df
df['Spending Score'] = w1*(df['Purchase Amount (USD)']/df['Purchase Amount (USD)'].max()) + w2*(df['Previous Purchases']/df['Previous Purchases'].max()) + w3*(df['Frequency per Year']/df['Frequency per Year'].max())

# 📌 เลือกคอลัมน์สำหรับ Clustering
X = df[['Age', 'Spending Score']].values

# 📌 ปรับสเกลข้อมูลให้เหมาะสม
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 หาจำนวนคลัสเตอร์ที่เหมาะสมโดยใช้ Silhouette Score
best_score = -1
best_k = 2  # เริ่มต้นที่ k = 2
for k in range(2, 10):  # ทดลองใช้จำนวนคลัสเตอร์ตั้งแต่ 2 ถึง 9
    kmedoids = KMedoids(n_clusters=k,metric="euclidean", random_state=42)
    labels = kmedoids.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    if score > best_score:
        best_score = score
        best_k = k

print(f"เลือกจำนวนคลัสเตอร์ที่เหมาะสมที่สุด: k = {best_k} (Silhouette Score: {best_score:.3f})")

# 📌 ใช้ KMedoids กับจำนวนคลัสเตอร์ที่เหมาะสม
kmedoids = KMedoids(n_clusters=best_k,metric="euclidean", random_state=42)
df['Cluster'] = kmedoids.fit_predict(X_scaled)

