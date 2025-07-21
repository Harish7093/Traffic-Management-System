# 🚦 Smart Traffic Management System using YOLO, ESP32-CAM & Streamlit

> **“In a world gridlocked by chaos, let intelligent signals breathe order.”**

Welcome to the **Smart Traffic Management System**, a real-time, AI + IoT-powered traffic control system that uses **YOLO object detection, ESP32-CAM live feeds, and a Streamlit dashboard** to dynamically manage traffic signals based on vehicle density and categories.

---

## 🛠️ Project Overview

This project aims to:
- **Detect vehicles in real-time** (cars, bikes, buses, trucks) using YOLO.
- **Estimate vehicle density** and classify congestion levels.
- **Adjust traffic signal timings dynamically** to reduce wait times.
- Stream live video and analytics on a **Streamlit dashboard**.
- **Control traffic LEDs via Arduino** connected through serial communication.
- Replace traditional manual signal timers with **AI-based adaptive control**, optimizing traffic flow and cutting fuel wastage.

---

## 🖥️ Tech Stack

- **YOLOv5** for object detection.
- **ESP32-CAM** for live video streaming.
- **OpenCV** for frame processing.
- **Streamlit** for the real-time dashboard.
- **Python** for backend logic.
- **Arduino Uno** for hardware control.
- **PySerial** for PC-to-Arduino communication.
- (Optional) **PCB + Relay modules** for real junction deployment.

---

## 🚀 Features

✅ Real-time multi-class vehicle detection.  
✅ Traffic density estimation.  
✅ Adaptive signal timing based on live data.  
✅ Wireless video feed using ESP32-CAM.  
✅ Streamlit-based control dashboard.  
✅ Expandable and modular codebase for future upgrades.

---

## 📸 Screenshots

_Add screenshots here once available:_
- YOLO detection output with bounding boxes.
- Streamlit dashboard interface.
- Photo of the hardware setup (ESP32-CAM + Arduino + signal LEDs).

---

## ⚙️ How It Works

1. **ESP32-CAM** streams live video to the YOLO pipeline.
2. **YOLO** detects vehicles in each frame and classifies them.
3. **Density calculation**:
   - Low → Green for 20s
   - Medium → Green for 40s
   - High → Green for 60s
   - Very High → Green for 90s
4. Signal timings are sent to **Arduino** over serial to control the traffic lights.
5. **Streamlit** displays the live feed, detection data, and control options.

---

## 🪛 Installation & Setup

1. **Clone the repo**
   ```bash
      git clone https://github.com/yourusername/smart-traffic-management.git
      cd smart-traffic-management


2. **install required libraries**

```
      pip install -r requirements.txt
```
3. **navigating to cd webapp using**

   ``` bash
      cd webapp
   ```
4. **running app**

   ```bash
      streamlit run Introduction.py
   ```

if any queries , feel free to contact

--> instagram
-->linkedIn
      links in profile 
