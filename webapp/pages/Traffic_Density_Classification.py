from cmath import rect
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import torch
from utils import classify_img, get_density_model, get_category_model, detection_img
import time

st.title("Traffic Density Classification")

classes = ['Empty', 'High', 'Low', 'Medium', 'Traffic Jam']
category_classes = ["background", "Auto", "Bus", "Car", "LCV", "Motorcycle", "Truck", "Tractor", "Multi-Axle"]

# Initialize both models
density_model = get_density_model()
category_model = get_category_model()
density_model.eval()
category_model.eval()

# Directory paths for default images
dirname = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(dirname, os.pardir)
density_img_dir = os.path.join(root_dir, "images", "density_img")

# ESP32-CAM URL
esp32_cam_url = "http://192.168.213.78:81/stream"  # Updated stream URL

# Confidence thresholds
conf_threshold = float(st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05))
iou_threshold = float(st.slider("IOU Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05))

# --- Only process on upload/capture, show YOLO vehicle boundaries, count, and density label ---

st.subheader("Upload or Capture Image for Vehicle Density Detection")
file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
camera_file = st.camera_input("Capture Image")
process_button = st.button("Process Image")

def vehicle_count_to_density_label(vehicle_count):
    if vehicle_count <= 2:
        return "Empty"
    elif vehicle_count <= 5:
        return "Low"
    elif vehicle_count <= 10:
        return "Medium"
    elif vehicle_count <= 20:
        return "High"
    else:
        return "Traffic Jam"

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
        idx = np.random.choice(range(5), 1)[0]
        default_img_path = os.path.join(density_img_dir, f"{idx}.jpg")
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
        # Perform YOLO vehicle detection
        detected_img, vehicle_count = detection_img(category_model, img_resized, category_classes, conf_threshold, iou_threshold)
        # Convert detected image to RGB for display
        detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
        # Get density label
        simple_density = vehicle_count_to_density_label(vehicle_count)
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image")
            st.image(img_resized, use_container_width=True)
        with col2:
            st.write("YOLO Detection Result")
            st.image(detected_img_rgb, use_container_width=True)
        st.success(f"Vehicle Count: {vehicle_count}")
        st.success(f"Traffic Density (by count): {simple_density}")
    except Exception as e:
        st.error(f"Error in detection: {str(e)}")

st.subheader("ESP32-CAM Vehicle Density Detection")
esp32_cam_url = "http://192.168.213.78:81/stream"  # Update if needed
esp32_button = st.button("Capture from ESP32-CAM")

if esp32_button:
    try:
        cap = cv2.VideoCapture(esp32_cam_url)
        if not cap.isOpened():
            st.error("Failed to open ESP32-CAM stream. Please check the connection.")
        else:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Try multiple times to get a valid frame
            max_attempts = 5
            for attempt in range(max_attempts):
                ret, frame = cap.read()
                if ret and frame is not None and not np.all(frame == 0):
                    break
                time.sleep(0.5)
            if ret and frame is not None and not np.all(frame == 0):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (480, 480))
                detected_img, vehicle_count = detection_img(category_model, frame_resized, category_classes, conf_threshold, iou_threshold)
                detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                simple_density = vehicle_count_to_density_label(vehicle_count)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("ESP32-CAM Frame")
                    st.image(frame_resized, use_container_width=True)
                with col2:
                    st.write("YOLO Detection Result")
                    st.image(detected_img_rgb, use_container_width=True)
                st.success(f"Vehicle Count: {vehicle_count}")
                st.success(f"Traffic Density (by count): {simple_density}")
            else:
                st.error("Could not capture a valid frame from ESP32-CAM after multiple attempts")
            cap.release()
            cv2.destroyAllWindows()
    except Exception as e:
        st.error(f"Error connecting to ESP32-CAM: {str(e)}")

# ... remove/skip all other detection, live, and model probability/classification UI ...