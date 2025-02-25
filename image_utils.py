# image_utils.py
def get_image_url(item):
    # Dictionary mapping items to their respective image URLs
    image_urls = {
        "Dress": "https://zapaka.com/cdn/shop/products/04033012-Purplefirst.jpg?v=1628925127",
        "Clothing": "https://www.thoughtco.com/thmb/ctxxtfGGeK5f_-S3f8J-jbY-Gp8=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/close-up-of-clothes-hanging-in-row-739240657-5a78b11f8e1b6e003715c0ec.jpg",
        "Jewelry": "https://www.buccellati.com/media/catalog/category/1_High_Jewelry.jpg",
        "Belt": "https://wildrhinoshoes.com.au/cdn/shop/files/Belts_1.jpg?v=1708316630",
        "Shirt": "https://www.mrporter.com/variants/images/1647597332681244/pr/w1000.jpg",
        # Add more mappings as needed
    }
    return image_urls.get(item, "https://example.com/default.jpg")  # Default image if not found

def display_image(st, item, width=1080):
    """Display an image given an item category using Streamlit."""
    url = get_image_url(item)
    st.image(url, caption=item, width=width)
