import streamlit as st
import tensorflow as tf
import numpy as np

# Function to predict the class of an uploaded image
def predict_class(test_image):
    model = tf.keras.models.load_model("Fruits_and_vegetables_recognition.h5")
    img = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Set the page configuration
st.set_page_config(page_title="Fruits & Vegetables Recognition", layout="wide", page_icon="🍎")

# Sidebar with Logo and Navigation

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction", "About Project"])

# Home Page
if app_mode == "Home":
    st.markdown("<h1 style='text-align: center; color: green;'>Fruits & Vegetables Recognition System</h1>", unsafe_allow_html=True)
    
    try:
        st.image("home_image.jpg", use_column_width=True)  # Make sure the home image exists
    except Exception:
        st.warning("Home image not found!")
    
    st.markdown("""
    <div style="padding: 20px; background-color: #f9f9f9; border-radius: 10px; margin-bottom: 20px;">
        <h2>Welcome to the Fruits & Vegetables Recognition System</h2>
        <p>Identify fruits and vegetables with the power of AI! Upload an image and let our advanced machine learning model recognize the items for you.</p>
    </div>
    """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("""
    <div style="background-color: #e9f7ef; padding: 15px; border-radius: 10px;">
        <h3>How It Works</h3>
        <ol>
            <li>Go to the Prediction page from the Dashboard.</li>
            <li>Upload an image of fruits or vegetables.</li>
            <li>Our AI model analyzes the image.</li>
            <li>Get instant results with the names of the recognized items.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# About Project Page
if app_mode == "About Project":
    st.title("About the Project")
    
    st.markdown("""
    <div style="padding: 20px; background-color: #f3f3f3; border-radius: 10px; margin-bottom: 20px;">
        <h2>Purpose of the Project</h2>
        <p>This system helps users identify fruits and vegetables using image recognition technology. It's designed for grocery item recognition, assisting in inventory management in supermarkets and farms.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Technology Stack")
    st.markdown("""
    <ul>
        <li><strong>Streamlit</strong>: For the web interface.</li>
        <li><strong>Keras & TensorFlow</strong>: For building the CNN model.</li>
        <li><strong>Python</strong>: Core development language.</li>
        <li><strong>OpenCV</strong>: For image processing.</li>
        <li><strong>NumPy & Pandas</strong>: Data manipulation tools.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.subheader("How the Model Works")
    st.markdown("""
    <ol>
        <li><strong>Image Preprocessing</strong>: Resizes, normalizes, and formats the uploaded image.</li>
        <li><strong>Feature Extraction</strong>: The CNN model identifies features like shape, color, and texture.</li>
        <li><strong>Classification</strong>: The model predicts the fruit or vegetable in the image.</li>
    </ol>
    """)

# Prediction Page
if app_mode == "Prediction":
    st.title("Prediction Page")
    st.markdown("<h3>Upload an image of a fruit or vegetable</h3>", unsafe_allow_html=True)
    
    test_image = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
    
    if test_image:
        st.image(test_image, use_column_width=True)
        
        if st.button("Predict"):
            result_index = predict_class(test_image)
            class_list = [
                'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
                'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
                'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 
                'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 
                'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
            ]
            st.success(f"Model has predicted: **{class_list[result_index]}**")

