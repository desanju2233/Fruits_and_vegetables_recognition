import streamlit as st
import tensorflow as tf
import numpy as np

def predict_class(test_image):
    model = tf.keras.models.load_model("Fruits_and_vegetables_recognition.h5")
    img = tf.keras.preprocessing.image.load_img(test_image,target_size = (128,128));
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr]) #CONVERT SINGLE IMAGE TO THE BATCH.
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","Prediction","About Project"])

#Main Page

if(app_mode == "Home"):
    st.header("FRUITS AND VEGETABLE RECOGNITIION SYSTEM")
    image_path = "home_image.jpg"
    st.image(image_path)
    

    # 1. Welcome Section
    st.title("Welcome to the Fruits & Vegetables Recognition System")
    st.write("""
    Identify fruits and vegetables with the power of AI! Upload an image and let our advanced machine learning model recognize the items for you.
    """)

    # 2. How It Works Section
    st.subheader("How It Works")
    st.write("""
    1. Go to the prediction page from the Dashboard.
    2. Upload an image of fruits or vegetables.
    3. Our AI model analyzes the image.
    4. Get instant results with the names of the recognized items.
    """)
    
if(app_mode == "About Project"):
    st.title("About the Project")

    st.subheader("Purpose of the Project")
    st.write("""
This fruits and vegetables recognition system is designed to help users quickly identify different types of produce using image recognition technology. The system leverages machine learning to classify various fruits and vegetables in real-time from images.

The main goal of this project is to demonstrate the power of AI in automating everyday tasks like recognizing grocery items or assisting in inventory management in supermarkets and farms.
""")
    st.subheader("Technology Stack")
    st.write("""
The system is built using the following technologies:

- **Streamlit**: For building the web application interface.
- **Keras with TensorFlow**: For creating the Convolutional Neural Network (CNN) model.
- **Python**: The primary language used for model training and application development.
- **OpenCV**: For image preprocessing and manipulation.
- **NumPy & Pandas**: For data handling and manipulation.
""")
    st.subheader("How the Model Works")
    st.write("""
The recognition system is powered by a Convolutional Neural Network (CNN) model trained on a dataset of images containing various fruits and vegetables.

The process works as follows:

1. **Image Preprocessing**: Each uploaded image is resized, normalized, and transformed into a format suitable for the model.
2. **Feature Extraction**: The CNN model identifies unique features of the image, such as shapes, colors, and textures.
3. **Classification**: Based on these features, the model predicts the most likely fruit or vegetable present in the image.
""")
    
#predictions
if(app_mode == "Prediction"):
    st.header("Prediction Page")
    test_image = st.file_uploader("Choose an Image")
    if(st.button("Show Image")):
        st.image(test_image)
    if(st.button("Predict")):
        result_index = predict_class(test_image)
        class_list = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
        st.write("Model has predicted :", format(class_list[result_index]))
        
    
