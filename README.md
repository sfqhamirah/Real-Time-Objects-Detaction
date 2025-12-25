# 🎯 Real-Time Object Detection

This project demonstrates **real-time object detection** using **YOLOv3 and YOLOv8** with **OpenCV** in Python.
It supports both **webcam live detection** and **video file detection**, making it suitable for learning, demos, and academic projects.

---

## 🚀 Features

* Real-time object detection using webcam
* Object detection on video files
* YOLOv3 (Darknet) implementation
* YOLOv8 (Ultralytics) implementation
* COCO dataset classes
* Bounding boxes with class labels and confidence scores

---

## 📁 Project Structure

```
Real-Time-Objects-Detection/
│
├── coco.names              # COCO class labels
├── real_time_yolov3.py     # YOLOv3 real-time detection script
├── real_time_yolov8.py     # YOLOv8 real-time detection script
├── walking.mp4             # Sample video for testing
├── yolov3.txt              # YOLOv3 notes
├── yolov8.txt              # YOLOv8 notes
├── yolov8n.pt              # Pre-trained YOLOv8 nano model
└── README.md               # Project documentation
```

---

## 🛠️ Requirements

Make sure Python is installed, then install the dependencies:

### For YOLOv8

```bash
pip install ultralytics opencv-python
```

### For YOLOv3

```bash
pip install opencv-python numpy
```

---

## ▶️ How to Run

### 🔴 YOLOv8 (Webcam)

```bash
python real_time_yolov8.py
```

### 🎥 YOLOv8 (Video File)

Update this line in the script:

```python
cap = cv2.VideoCapture("walking.mp4")
```

---

### 🔵 YOLOv3 (Webcam or Video)

```bash
python real_time_yolov3.py
```

Make sure all YOLOv3 files (weights, config, coco.names) are correctly linked in the script.

---

## 🧠 Model Details

* **YOLOv3**

  * Accurate but heavier
  * Slower on CPU

* **YOLOv8 (Nano)**

  * Lightweight and fast
  * Optimized for real-time performance
  * Ideal for CPU-based systems

---

## 📌 Dataset

This project uses the **COCO dataset**, which includes 80 object classes such as:

* Person
* Car
* Bicycle
* Dog
* Chair
* Laptop

---

## 📷 Output Example

* Bounding boxes drawn around detected objects
* Class name and confidence score displayed
* Real-time FPS performance

---

## 💡 Future Improvements

* Add FPS counter
* Custom dataset training
* Object tracking integration
* Deploy as a web app using Streamlit

---
