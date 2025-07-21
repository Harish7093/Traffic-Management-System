import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import torch
import time
from utils import get_category_model, get_density_model, detection_img, classify_img
import serial

st.title("Combined Vehicle Detection with Traffic Signal Control")

# Traffic Signal Configuration
SIGNAL_COLORS = ["red", "green", "yellow"]
# Adjust these durations to fit within 30 secondcds total
CYCLE_DURATION = 30  # Total cycle time in seconds
BASE_DURATIONS = {
    "red": 13,     # red duration
    "green": 13,   # green duration
    "yellow": 4    # yellow duration
}
# Verify the total equals our cycle time
assert sum(BASE_DURATIONS.values()) == CYCLE_DURATION, "Signal durations must sum to cycle time"

# Initialize signal state
if 'signal_state' not in st.session_state:
    st.session_state.signal_state = {
        "Traffic Light 1": {
            "color": "red",
            "time_left": BASE_DURATIONS["red"],
            "vehicle_count": 0
        },
        "Traffic Light 2": {
            "color": "red",
            "time_left": BASE_DURATIONS["red"],
            "vehicle_count": 0
        }
    }
    st.session_state.current_light = "Traffic Light 1"
    st.session_state.last_signal_update = time.time()
    st.session_state.cycle_completed = False
    # Initialize captured images storage
    st.session_state.captured_images = {
        "Traffic Light 1": [],
        "Traffic Light 2": []
    }
    st.session_state.last_capture_time = {
        "Traffic Light 1": 0,
        "Traffic Light 2": 0
    }

# Initialize serial connection
try:
    ser = serial.Serial('COM3', 9600)  # Adjust COM port as needed
except:
    ser = None
    st.warning("Could not connect to Arduino. Simulating signals instead.")

def send_command_to_arduino(command):
    """
    Commands format:
    - First character: Color (R, Y, G)
    - Second character: Light number (1, 2)
    Example: 'R1' = Red for Traffic Light 1, 'G2' = Green for Traffic Light 2
    """
    if ser:
        try:
            # Turn off all lights first
            ser.write(f"O{command[1]}".encode())  # O for Off
            time.sleep(0.1)  # Small delay
            
            # Send the actual command
            ser.write(command.encode())
            time.sleep(0.1)  # Small delay to ensure command is processed
        except Exception as e:
            st.error(f"Failed to send command to Arduino: {str(e)}")
    else:
        st.info(f"Would send {command} to Arduino")

# Add CSS for traffic light
st.markdown("""
    <style>
    .traffic-light {
        width: 120px;
        height: 350px;
        background-color: #333;
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        align-items: center;
        padding: 15px;
        margin: 20px auto;
    }
    .traffic-light .light {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin: 10px;
        transition: all 0.3s ease;
    }
    .traffic-light .light.active {
        box-shadow: 0 0 20px currentColor;
    }
    </style>
""", unsafe_allow_html=True)

# Function to display traffic signal using a placeholder
def display_signal(placeholder):
    # Create columns for the two traffic lights
    col1, col2 = placeholder.columns(2)
    
    for idx, light_name in enumerate(["Traffic Light 1", "Traffic Light 2"]):
        # Get current signal state for this light
        color = st.session_state.signal_state[light_name]["color"]
        time_left = int(st.session_state.signal_state[light_name]["time_left"])
        vehicle_count = st.session_state.signal_state[light_name]["vehicle_count"]
        
        # Use the appropriate column
        col = col1 if idx == 0 else col2
        
        with col:
            st.markdown(f"<h3 style='text-align: center;'>{light_name}</h3>", unsafe_allow_html=True)
            
            # Container for traffic light
            st.markdown("""
                <div class="traffic-light">
                    <div class="light {}" style="background-color: red; opacity: {}"></div>
                    <div class="light {}" style="background-color: yellow; opacity: {}"></div>
                    <div class="light {}" style="background-color: green; opacity: {}"></div>
                </div>
            """.format(
                'active' if color == 'red' else '', '1' if color == 'red' else '0.3',
                'active' if color == 'yellow' else '', '1' if color == 'yellow' else '0.3',
                'active' if color == 'green' else '', '1' if color == 'green' else '0.3'
            ), unsafe_allow_html=True)
            
            # Display time and vehicle count
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div style='font-size: 24px; color: #4CAF50;'>{time_left}s</div>
                    <div style='color: #666;'>Vehicles Waiting</div>
                    <div style='font-size: 20px; color: #2196F3;'>{vehicle_count}</div>
                    <div style='font-size: 18px; color: #FF5722;'>
                        {'Active' if light_name == st.session_state.current_light else 'Waiting'}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Function to calculate dynamic durations based on vehicle count
def calculate_durations(vehicle_count):
    total_cycle = 30  # Fixed at 30 seconds
    yellow_time = 5   # Fixed yellow duration
    
    # Adjust red and green based on traffic but maintain 30-second total
    if vehicle_count > 10:  # High traffic - favor green
        green_time = 17
        red_time = total_cycle - yellow_time - green_time
    elif vehicle_count < 3:  # Low traffic - favor red
        green_time = 8
        red_time = total_cycle - yellow_time - green_time
    else:  # Medium traffic - balanced
        green_time = 13
        red_time = total_cycle - yellow_time - green_time
    
    return {
        "red": red_time,
        "yellow": yellow_time,
        "green": green_time
    }

# Function to capture and store image
def capture_and_store_image(frame, light_name):
    current_time = time.time()
    # Only capture if enough time has passed since last capture (e.g., 5 seconds)
    if current_time - st.session_state.last_capture_time[light_name] >= 5:
        # Store the image
        st.session_state.captured_images[light_name].append(frame.copy())
        st.session_state.last_capture_time[light_name] = current_time
        # Keep only the last 5 captured images
        if len(st.session_state.captured_images[light_name]) > 5:
            st.session_state.captured_images[light_name].pop(0)
        return True
    return False

# Function to clear captured frames
def clear_captured_frames():
    st.session_state.captured_images = {
        "Traffic Light 1": [],
        "Traffic Light 2": []
    }
    st.session_state.last_capture_time = {
        "Traffic Light 1": 0,
        "Traffic Light 2": 0
    }

# Function to clear captured frames for a specific light
def clear_captured_frames_for_light(light_name):
    st.session_state.captured_images[light_name] = []
    st.session_state.last_capture_time[light_name] = 0

# Function to process a complete signal cycle
def process_signal_cycle(vehicle_count, density_class, signal_placeholder, status_text):
    clear_captured_frames_for_light(st.session_state.current_light)
    current_light = st.session_state.current_light
    next_light = "Traffic Light 2" if current_light == "Traffic Light 1" else "Traffic Light 1"
    light_number = "1" if current_light == "Traffic Light 1" else "2"
    other_light_number = "2" if current_light == "Traffic Light 1" else "1"
    
    # Get custom durations based on traffic
    durations = calculate_durations(vehicle_count)
    
    # Store the vehicle count for display
    st.session_state.signal_state[current_light]["vehicle_count"] = vehicle_count
    
    # Sequence: Red → Yellow → Green
    sequence = [
        {"color": "red", "duration": durations["red"], "arduino_cmd": "R"},
        {"color": "yellow", "duration": durations["yellow"], "arduino_cmd": "Y"},
        {"color": "green", "duration": durations["green"], "arduino_cmd": "G"}
    ]
    
    # Run through each phase in the sequence
    for phase in sequence:
        color = phase["color"]
        duration = phase["duration"]
        arduino_cmd = phase["arduino_cmd"]
        
        # Send command to Arduino for the current light
        send_command_to_arduino(f"{arduino_cmd}{light_number}")
        time.sleep(0.1)  # Small delay to ensure command is processed
        
        # Set the signal state for this phase
        st.session_state.signal_state[current_light]["color"] = color
        st.session_state.signal_state[current_light]["time_left"] = duration
        
        # Set the opposite color for the other light
        opposite_color = "green" if color == "red" else "red"
        st.session_state.signal_state[next_light]["color"] = opposite_color
        st.session_state.signal_state[next_light]["time_left"] = duration
        
        # Send command to Arduino for the other light
        opposite_cmd = "G" if color == "red" else "R"
        send_command_to_arduino(f"{opposite_cmd}{other_light_number}")
        
        # Update status text
        status_text.info(f"{current_light} - {color.capitalize()} light - {duration}s")
        
        # Update display every second for this phase
        phase_start = time.time()
        while time.time() - phase_start < duration:
            elapsed = time.time() - phase_start
            st.session_state.signal_state[current_light]["time_left"] = max(0, duration - elapsed)
            display_signal(signal_placeholder)
            time.sleep(1)
        
        # Add a 2-second delay after the green light phase
        if color == "green":
            time.sleep(2)
            # Capture a new image after the delay
            ret, frame = cap.read()
            if ret:
                capture_and_store_image(frame, current_light)
    
    # Switch to the next traffic light
    st.session_state.current_light = next_light
    
    display_signal(signal_placeholder)
    status_text.success(f"{current_light} cycle completed. Switching to {next_light}")
    
    return sum(phase["duration"] for phase in sequence)

# Initialize models
category_classes = ["background", "Auto", "Bus", "Car", "LCV", "Motorcycle", "Truck", "Tractor", "Multi-Axle"]
density_classes = ['Empty', 'High', 'Low', 'Medium', 'Traffic Jam']

category_model = get_category_model()
density_model = get_density_model()
category_model.eval()
density_model.eval()

# Directory paths for default images
dirname = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(dirname, os.pardir)
default_img_dir = os.path.join(root_dir, "images", "default_img")

# Slider widgets for setting confidence and IOU thresholds
c1, c2 = st.columns(2)
with c1:
    conf_threshold = float(st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.03, step=0.02))
with c2:
    iou_threshold = float(st.slider("IOU Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.02))

# File uploader widget for uploading images
file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

# Camera input widget for taking a photo
camera_file = st.camera_input("Capture Image")

# ESP32-CAM URL
esp32_cam_url = "http://192.168.213.78:81/stream"

# Hardware button for ESP32-CAM
hardware_button = st.button("Hardware")

# Create placeholder for the traffic light
signal_placeholder = st.empty()
status_text = st.empty()

# Display initial traffic light
display_signal(signal_placeholder)

if hardware_button:
    stframe = st.empty()
    prob_frame = st.empty()
    captured_images_container = st.empty()
    try:
        cap = cv2.VideoCapture(esp32_cam_url)
        if not cap.isOpened():
            st.error("Failed to open ESP32-CAM stream. Please check the connection and make sure the camera is streaming.")
        else:
            while True:
                # Capture and analyze frame immediately
                st.info("Capturing and analyzing frame...")
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                frame_resized = cv2.resize(frame, (480, 480))
                
                # Perform both detections
                category_img, vehicle_count = detection_img(category_model, frame_resized, category_classes, conf_threshold, iou_threshold)
                density_label, density_prob = classify_img(density_model, frame_resized)
                
                # Display detection results
                stframe.image(category_img, channels="BGR", use_column_width=True)
                prob_frame.success(
                    f"Analysis Complete:\n\n"
                    f"\n  Density: {density_classes[density_label]} (Probability: {density_prob:.4f})\n\n"
                    f"\n  Vehicle Count: {vehicle_count}\n"
                    f"\n  Processing traffic light cycle\n"
                )
                
                # Capture and store image before processing the signal cycle
                if capture_and_store_image(frame_resized, st.session_state.current_light):
                    # Display captured images
                    with captured_images_container.container():
                        st.subheader(f"Captured Images for {st.session_state.current_light}")
                        cols = st.columns(min(5, len(st.session_state.captured_images[st.session_state.current_light])))
                        for idx, img in enumerate(st.session_state.captured_images[st.session_state.current_light]):
                            with cols[idx]:
                                st.image(img, channels="BGR", use_column_width=True)
                
                # Process the complete signal cycle
                process_signal_cycle(vehicle_count, density_classes[density_label], signal_placeholder, status_text)
                
                # Check if we should stop
                if not st.session_state.get("hardware_running", True):
                    if ser:
                        ser.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    break
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Button to start detection (image or live)
button = st.button("Detect")

if button:
    if file is not None:
        # Image uploaded
        title = "Uploaded Image"
        img = Image.open(file)
        img = np.array(img)
        img = cv2.resize(img, (480, 480))
    elif camera_file is not None:
        # Image taken from camera
        title = "Captured Image"
        img = Image.open(camera_file)
        img = np.array(img)
        img = cv2.resize(img, (480, 480))
    else:
        # Default image if no file or camera input
        title = "Default Image"
        idx = np.random.choice(range(5), 1)[0]
        default_img_path = os.path.join(default_img_dir, f"{idx}.jpg")
        img = Image.open(default_img_path)
        img = np.array(img)
        img = cv2.resize(img, (480, 480))

    # Perform both detections
    category_img, vehicle_count = detection_img(category_model, img, category_classes, conf_threshold, iou_threshold)
    density_label, density_prob = classify_img(density_model, img)

    # Display the results
    col1, col2 = st.columns(2)
    with col1:
        st.write(title)
        st.image(img)
    with col2:
        st.write("Vehicle Detection Results")
        st.image(category_img)
        st.success(f"Density Classification: {density_classes[density_label]}")
        st.info(f"Probability: {density_prob:.4f}")
        st.success(f"Total Vehicles Detected: {vehicle_count}")
    
    # Update the traffic light with the vehicle count
    st.session_state.signal_state["Traffic Light 1"]["vehicle_count"] = vehicle_count
    display_signal(signal_placeholder)

# Real-time live detection with complete cycle processing
live_button_key = "live_button"
if st.button("Start Automatic Analysis", key=live_button_key):
    stframe = st.empty()
    prob_frame = st.empty()
    count_frame = st.empty()
    
    # Initialize webcam outside the loop
    cap = cv2.VideoCapture(1)  # Capture from webcam
    if not cap.isOpened():
        st.error("Failed to open camera")
    else:
        try:
            while True:
                # Capture and analyze frame immediately
                st.info("Capturing and analyzing frame...")
                
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                frame_resized = cv2.resize(frame, (480, 480))
                category_img, vehicle_count = detection_img(category_model, frame_resized, category_classes, conf_threshold, iou_threshold)
                density_label, density_prob = classify_img(density_model, frame_resized)
                
                # Display detection results
                stframe.image(category_img, channels="BGR", use_column_width=True)
                count_frame.success(f"Vehicles Detected: {vehicle_count}")
                prob_frame.success(
                    f"Analysis Complete:\n"
                    f"Density: {density_classes[density_label]} (Probability: {density_prob:.4f})\n"
                    f"Processing traffic light cycle"
                )
                
                # Process the complete traffic signal cycle
                process_signal_cycle(vehicle_count, density_classes[density_label], signal_placeholder, status_text)
                
                # Check if we should stop
                if not st.session_state.get(live_button_key, False):
                    break
        finally:
            # Always release the webcam when done
            cap.release()
            cv2.destroyAllWindows()