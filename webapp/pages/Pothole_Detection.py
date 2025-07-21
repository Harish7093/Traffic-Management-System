import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import torch
import time
from utils import get_pothole_model, detection_img

# Initialize session state for live stream control
if "hardware_running" not in st.session_state:
    st.session_state.hardware_running = False
if "live_detection" not in st.session_state:
    st.session_state.live_detection = False

st.title("Pothole Object Detection")

# Model and class setup
classes = ["Background", "Pothole"]
dirname = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(dirname, os.pardir)
pothole_img_dir = os.path.join(root_dir, "images", "pothole_img")

model = get_pothole_model()
model.eval()

# Upload or capture an image
file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
camera_file = st.camera_input("Capture Image")

# Threshold sliders
c1, c2 = st.columns(2)
with c1:
    conf_threshold = float(st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.02))
with c2:
    iou_threshold = float(st.slider("IOU Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.02))

col1, col2 = st.columns(2)
button = st.button("Detect")

# ESP32-CAM Stream (Single Frame Detection)
esp32_cam_url = "http://192.168.213.78:81/stream"  # Replace with your ESP32-CAM stream URL
hardware_button = st.toggle("Start ESP32-CAM Detection")

if hardware_button:
    st.session_state.hardware_running = True
    stframe = st.empty()

    cap = cv2.VideoCapture(esp32_cam_url)
    if not cap.isOpened():
        st.error("Failed to open ESP32-CAM stream.")
    else:
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (480, 480))
            detect_img, num_detections = detection_img(model, frame_resized, classes, conf_threshold, iou_threshold)
            stframe.image(detect_img, channels="BGR", use_column_width=True)
            stframe.success(f"Detected {num_detections} potholes")
        else:
            st.error("Could not read frame from ESP32-CAM.")
        cap.release()
else:
    st.session_state.hardware_running = False

# Image detection from upload or camera
if button:
    if file is not None:
        title = "Uploaded Image"
        img = Image.open(file)
    elif camera_file is not None:
        title = "Captured Image"
        img = Image.open(camera_file)
    else:
        title = "Default Image"
        idx = np.random.choice(range(4), 1)[0]
        default_img_path = os.path.join(pothole_img_dir, f"{idx}.png")
        img = Image.open(default_img_path)

    img = np.array(img)
    img = cv2.resize(img, (480, 480))

    with col1:
        st.write(title)
        st.image(img, channels="RGB")

    with col2:
        st.write("Detection Output")
        detect_img, num_detections = detection_img(model, img, classes, conf_threshold, iou_threshold)
        st.image(detect_img, channels="BGR")
        st.success(f"Detected {num_detections} potholes")

# Webcam-based Live Detection
live_detection = st.toggle("Start Live Detection with Webcam")

if live_detection:
    st.session_state.live_detection = True
    stframe = st.empty()
    prob_frame = st.empty()

    cap = cv2.VideoCapture()
    if not cap.isOpened():
        st.error("Unable to access the webcam.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            frame_resized = cv2.resize(frame, (480, 480))
            detect_img, num_detections = detection_img(model, frame_resized, classes, conf_threshold, iou_threshold)
            
            # Display the current frame with detection
            stframe.image(detect_img, channels="BGR", use_column_width=True)
            prob_frame.success(f"Detected {num_detections} potholes")
            
            # Check if we should stop
            if not st.session_state.get("live_detection", True):
                cap.release()
                cv2.destroyAllWindows()
                break
            
            time.sleep(0.1)  # Small delay to prevent overwhelming the system
else:
    st.session_state.live_detection = False
