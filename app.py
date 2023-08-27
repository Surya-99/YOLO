import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from yolo import res
from base64 import b64encode

# Define the get_download_link function
def get_download_link(img_bytes):
    href = f'<a href="data:image/png;base64,{b64encode(img_bytes).decode()}" download="processed_image.png">Click here to download the processed image</a>'
    return href

# Streamlit UI
st.title("Picture Upload and Submit")

# Upload image
uploaded_image = st.file_uploader("Upload a picture", type=["jpg", "png", "jpeg"])

# Display uploaded image
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Picture", use_column_width=True)

# Submit button
submit_button = st.button("Submit")

# Handle submit action
if submit_button:
    if uploaded_image is not None:
        # Convert uploaded image to numpy array
        image_array = np.array(Image.open(uploaded_image))
        
        # Perform object detection using the res() function
        annotated_image, dic = res(image_array)

        # Display annotated image
        st.image(annotated_image, caption=dic, use_column_width=True)
        
        # Convert annotated image to PNG format
        annotated_image_pil = Image.fromarray(annotated_image)
        image_io = BytesIO()
        annotated_image_pil.save(image_io, format="PNG")
        image_data = image_io.getvalue()
        
        # Generate and display download link
        download_link = get_download_link(image_data)
        st.markdown(download_link, unsafe_allow_html=True)

    else:
        st.warning("Please upload an image before submitting.")
