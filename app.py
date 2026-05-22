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

    st.image(image, caption="Uploaded Image", use_container_width=True)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    results = model.predict(
        source=temp_file.name,
        conf=0.25,
        save=True
    )

    output_path = results[0].save_dir
    output_image = os.path.join(
        output_path,
        os.path.basename(temp_file.name)
    )

    st.subheader("Detection Result")
    st.image(output_image, use_container_width=True)