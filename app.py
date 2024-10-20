import requests
import json
import numpy as np
import streamlit as st
import cv2
from PIL import Image

np.random.seed(42)

url =  'https://apexherbert200-mnist-digit-api.hf.space/predict'

array_ = np.zeros((784), dtype=int)


with st.sidebar:
   file = st.file_uploader(" ", type=["jpeg", "png", "jpg", "webp"])
   if file is not None:
        st.success("File has been uploaded :)")
   else:
        st.warning("No file has been uploaded :(")

if file is not None:
    st.markdown("# Digit Classifier ")
    st.info("An application to classify handwritten digits based on the image ")
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img_data = cv2.imdecode(file_bytes, 1)  # Decode the image
    st.image(img_data, channels="BGR", caption="Uploaded Image")
        
    gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    array_file = np.array(gray_img, dtype=np.float64)
   

    scale_img = cv2.resize(gray_img, (28,28), interpolation = cv2.INTER_AREA)

    flattened_img = scale_img.flatten()
   



    data = {
        "image_x":flattened_img.tolist()  # Example list of floats
    }
    if st.button("Predict Digit ❤️"):
        new_request = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    
        st.success(f" The number is {new_request.json()['prediction']}")
        st.markdown(f"##  {new_request.json()['prediction']}")

if file is None:
    st.markdown("# Digit Classifier ")
    st.info("An application to classify handwritten digits based on the image ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.warning("No file has been uploaded. Please upload a file ")
