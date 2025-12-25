from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model (nano = light CPU-friendly)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(3)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict objects
    results = model(frame)

    # Show results
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
