import cv2
import numpy as np
import streamlit as st

st.title("Shrimp Seed Counter")

# Streamlit file uploader
image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Convert uploaded file to OpenCV image format
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Display the original image
    st.image(img, caption='Original Image', use_column_width=True)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    thresh=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,29,10)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area (if necessary)
    shrimp_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0]

    # Draw contours on the original image
    cv2.drawContours(img, shrimp_contours, -1, (0, 255, 0), 1)

    # Count the number of contours
    shrimp_count = len(shrimp_contours)

    # Add text with shrimp count to the image
    cv2.putText(img, f'Shrimp count: {shrimp_count}', (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

    # Display the image with contours
    st.image(img, caption=f'Processed Image with Shrimp Count: {shrimp_count}', use_column_width=True)
