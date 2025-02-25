from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from PIL import Image

def preprocess_data(data):
    numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    categorical_features = ['Gender', 'Category', 'Item Purchased']
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor.fit_transform(data), preprocessor

def resize_image(image, base_width):
    w_percent = (base_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    return image.resize((base_width, h_size), Image.Resampling.LANCZOS)
