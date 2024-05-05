import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
from classifier import classify_image, set_background_picture

set_background("./background.jpg")

st.title("Pneumonia detector")

st.header("please upload your x-ray in order to detect if you have Pneumonia")
class_name = ["pneumonia","healthy"]
file_uploaded = st.file_uploader("",type=['jpg','jpeg','png'])

model_1 = load_model('./model/model_1.h5')


image = Image.open(file_uploaded).convert('RGB')
st.image(image, use_column_width=True)

class_name, conf_score = classify(image, model_1, class_name)

st.write("## {}".format(class_name))
st.write("### score: {}%".format(int(conf_score * 1000) / 10))