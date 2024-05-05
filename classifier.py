import base64
import numpy as np
import streamlit as st
from PIL import ImageOps, Image


def classify_image(image, model, class_names):

    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    
    normalized_image_array = (image_array.astype(np.float32) / 255)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)

    if prediction[0][0] > 0.75:
        index = 0
    else:
        index = 1
    class_name = class_names[index]
    linear_score = prediction[0][index]

    return class_name, linear_score



def set_background_picture(image_file):

    with open(image_file, "rb") as f:
        img_data = f.read()

    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

