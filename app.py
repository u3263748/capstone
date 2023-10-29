import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

st.markdown("# Logo or No-Go")
st.markdown("Provide a picture of a logo to determine whether or not it's genuine.")

model = load_model("keras_model.h5")
st.markdown("Model loaded and compiled with no errors.")

# Load the labels
class_names = open("labels.txt", "r").readlines()

def predict(image):
    # This function is a modified version of Teachable Machine's Keras export snippet.
    # Google Inc, (no date), Teachable Machine. https://teachablemachine.withgoogle.com/
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def display_status(class_name, score, image):
    st.image(image)
    if class_name == "Genuine":
        st.success(f"This logo may be genuine. The probability of this being the case is {score:.2f}.", icon="✅")
    elif class_name == "Fake":
        st.warning(f"This logo may not be genuine. The probability of this being the case is {score:.2f}.", icon="⚠️")
    else:
        # Shouldn't occur: there are only two possible classes of image.
        st.info(f'This logo is {class_name}, with a probability of {score:.2f}.')


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    class_name, score = predict(image)
    
    # Prediction returns the binary 0/1 value alongside the name of the predicted class,
    # so .split() is used to remove it. .strip() is also used to remove stray space characters appearing after the class.
    display_status(class_name.split(" ")[1].strip(), score, image)