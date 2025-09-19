# app.py

# Uncomment below if running in Google Colab or need to install packages
# !pip install streamlit opencv-python pillow torch torchvision torchaudio

# Clone YOLOv5 once, not every time
# !git clone https://github.com/ultralytics/yolov5.git

# Import libraries
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Append YOLOv5 repo to path if running from cloned repo
YOLO_DIR = 'yolov5'
if YOLO_DIR not in sys.path:
    sys.path.append(YOLO_DIR)

# Load model (make sure yolov5 directory exists)
model = torch.hub.load('D:/INTERNSHIP PROJECT1/yolov5', 'yolov5s', source='local')  # uses yolov5s model
model.eval()

# Streamlit UI
st.title("🚀 YOLOv5 Object Detection Web App")
st.write("Upload an image and see YOLOv5 detect objects in it.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# When file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Detect Objects"):
        # Convert PIL image to NumPy
        img_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Perform detection
        results = model(img_bgr)
        detections = results.pandas().xyxy[0]

        # Draw bounding boxes and labels
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_array, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show detected image
        st.image(img_array, caption='Detected Image', channels="RGB", use_column_width=True)
