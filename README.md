# 👣 Footfall Counter using Computer Vision

This project demonstrates a real-time **AI-powered footfall counter** that detects and tracks people entering and exiting through a defined area (doorway, corridor, or gate).

## 🎯 Objective
Count how many people **enter and exit** through a region using **YOLO + DeepSORT** tracking in a Streamlit web app.

---

## 🧠 Approach

### 1. Detection
- Uses **YOLOv8 (Ultralytics)** for human detection.

### 2. Tracking
- **DeepSORT** tracks individuals across frames using appearance + motion features.

### 3. Counting Logic
- A **virtual line** is drawn at the center.
- If a person crosses the line from **top to bottom → Entry**  
  From **bottom to top → Exit**

---

## 🖥️ Features
✅ Real-time video or file upload support  
✅ Live entry/exit count display  
✅ Works on webcam or uploaded videos  
✅ Clean, modern Streamlit UI  

---

## 🚀 How to Run

### 1️⃣ Setup Environment
```bash
python -m venv venf
venv\Scripts\activate
pip install -r requirements.txt
