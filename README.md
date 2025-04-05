Fruits & Vegetables Recognition System 🍎🥦
This project is a Fruits and Vegetables Recognition System built using TensorFlow, Keras, and Streamlit. It leverages a pre-trained Convolutional Neural Network (CNN) model to classify images of fruits and vegetables.

Demo : Have a demo by clicking this  link : [link text](https://fruitsandvegetablesrecognition-6gvxcmmkhxwcbijxcxhpaz.streamlit.app/)

# Features
- Upload an image of fruits or vegetables to get predictions.
- Uses a CNN model for image classification.
- User-friendly web interface built with Streamlit.
- Displays prediction results along with the uploaded image.
- Simple and responsive design.
- Technologies Used
- TensorFlow & Keras: For building the deep learning model.
- Streamlit: For the front-end interface.
- Python: As the core programming language.
- OpenCV: For image preprocessing (optional, based on your usage).
- NumPy: For numerical operations.
# Project Structure

- **Fruits_and_vegetables_recognition.h5**  # Trained model
- **main.py**                               # Streamlit app file
- **logo.png**                              # Optional logo for the sidebar
- **home_image.jpg**                        # Optional image for the home page
- **requirements.txt**                      # Python dependencies
- **README.md**                             # Project documentation
  
### How to Run the Project
Prerequisites
Python 3.8 or later.
Make sure you have pip installed.
Installation

## Install the required dependencies:

pip install -r requirements.txt
Ensure the trained model Fruits_and_vegetables_recognition.h5 is in the project directory.

### Run the App
After installing the dependencies, run the Streamlit app with the following command:

### streamlit run main.py
This will start a local web server and the app will be accessible at http://localhost:8501 in your browser.

### How It Works
- Home Page: Provides an overview of the app with a user-friendly introduction.
- Prediction Page: Allows users to upload an image. The system will display the uploaded image and predict the class (fruit/vegetable) using the CNN model.
- About Page: Contains details about the project, its purpose, and the technology stack.
### Dataset
The model was trained using a dataset of various fruits and vegetables images, which were resized and preprocessed for training.

### Model Details
The CNN model was trained using Keras and TensorFlow. It consists of several Conv2D, MaxPooling, and Dense layers for feature extraction and classification. The input images are resized to 128x128 pixels.
