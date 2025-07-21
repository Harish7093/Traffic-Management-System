import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import torch
from utils import get_category_model, detection_img
import time

st.title("Vehicle Category Detection")

classes = ["background", "Auto", "Bus", "Car", "LCV", "Motorcycle", "Truck", "Tractor", "Multi-Axle"]
model = get_category_model()
model.eval()

# Directory paths for default images
dirname = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(dirname, os.pardir)
category_img_dir = os.path.join(root_dir, "images", "category_img")

# ESP32-CAM URL
esp32_cam_url = "http://192.168.213.78:81/stream"  # Updated stream URL

# Slider widgets for setting confidence and IOU thresholds
c1, c2 = st.columns(2)
with c1:
    conf_threshold = float(st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.03, step=0.02))
with c2:
    iou_threshold = float(st.slider("IOU Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.02))

# --- Only process on upload/capture, show YOLO vehicle category boxes, count, and list ---
st.subheader("Upload or Capture Image for Vehicle Category Detection")
file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
camera_file = st.camera_input("Capture Image")
process_button = st.button("Process Image")

def get_category_list_from_detection(detection_result, class_names):
    # detection_result is the output image with boxes, but we need to get the list of detected categories
    # If detection_img returns (img, count, labels), use that. If not, just return count and empty list.
    if isinstance(detection_result, tuple) and len(detection_result) == 3:
        _, count, label_indices = detection_result
        detected_categories = [class_names[i] for i in set(label_indices) if i > 0]
        return count, detected_categories
    elif isinstance(detection_result, tuple) and len(detection_result) == 2:
        _, count = detection_result
        return count, []
    else:
        return 0, []

if process_button:
    if file is not None:
        title = "Uploaded Image"
        img = Image.open(file)
        img = np.array(img)
    elif camera_file is not None:
        title = "Captured Image"
        img = Image.open(camera_file)
        img = np.array(img)
    else:
        title = "Default Image"
        idx = np.random.choice(range(7), 1)[0]
        default_img_path = os.path.join(category_img_dir, f"{idx}.jpg")
        img = Image.open(default_img_path)
        img = np.array(img)

    # Convert to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    # Resize image
    img_resized = cv2.resize(img_rgb, (480, 480))

    try:
        # Perform YOLO vehicle category detection
        detection_result = detection_img(model, img_resized, classes, conf_threshold, iou_threshold)
        if isinstance(detection_result, tuple):
            detect_img = detection_result[0]
            vehicle_count, detected_categories = get_category_list_from_detection(detection_result, classes)
        else:
            detect_img = detection_result
            vehicle_count, detected_categories = 0, []
        detect_img_rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image")
            st.image(img_resized, use_container_width=True)
        with col2:
            st.write("YOLO Category Detection Result")
            st.image(detect_img_rgb, use_container_width=True)
        st.success(f"Vehicle Count: {vehicle_count}")
        if detected_categories:
            st.success(f"Categories Detected: {', '.join(detected_categories)}")
    except Exception as e:
        st.error(f"Error in detection: {str(e)}")

st.subheader("ESP32-CAM Vehicle Category Detection")
esp32_button = st.button("Capture from ESP32-CAM")

if esp32_button:
    try:
        cap = cv2.VideoCapture(esp32_cam_url)
        if not cap.isOpened():
            st.error("Failed to open ESP32-CAM stream. Please check the connection.")
        else:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            max_attempts = 5
            for attempt in range(max_attempts):
                ret, frame = cap.read()
                if ret and frame is not None and not np.all(frame == 0):
                    break
                time.sleep(0.5)
            if ret and frame is not None and not np.all(frame == 0):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (480, 480))
                detection_result = detection_img(model, frame_resized, classes, conf_threshold, iou_threshold)
                if isinstance(detection_result, tuple):
                    detect_img = detection_result[0]
                    vehicle_count, detected_categories = get_category_list_from_detection(detection_result, classes)
                else:
                    detect_img = detection_result
                    vehicle_count, detected_categories = 0, []
                detect_img_rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("ESP32-CAM Frame")
                    st.image(frame_resized, use_container_width=True)
                with col2:
                    st.write("YOLO Category Detection Result")
                    st.image(detect_img_rgb, use_container_width=True)
                st.success(f"Vehicle Count: {vehicle_count}")
                if detected_categories:
                    st.success(f"Categories Detected: {', '.join(detected_categories)}")
            else:
                st.error("Could not capture a valid frame from ESP32-CAM after multiple attempts")
            cap.release()
            cv2.destroyAllWindows()
    except Exception as e:
        st.error(f"Error connecting to ESP32-CAM: {str(e)}")