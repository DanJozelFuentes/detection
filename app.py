import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.title("Student Attention Detection")

model = YOLO("best.pt")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width='stretch')

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

    image.save(temp_file.name)

    results = model.predict(
        source=temp_file.name,
        conf=0.25
    )

    result_image = results[0].plot()

    st.subheader("Detection Result")

    st.image(result_image, width='stretch')