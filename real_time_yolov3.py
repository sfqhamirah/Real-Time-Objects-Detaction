import cv2
import numpy as np
import time
import os

# Base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load YOLOv3-tiny
net = cv2.dnn.readNet(
    os.path.join(BASE_DIR, "weights", "yolov3-tiny.weights"),
    os.path.join(BASE_DIR, "cfg", "yolov3-tiny.cfg")
)

# Load class names
classes = []
with open(os.path.join(BASE_DIR, "coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Random colors for bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Open webcam
cap = cv2.VideoCapture(3)

font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    height, width, _ = frame.shape

    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Detect objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4:  # detection threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

    # Draw bounding boxes
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            conf = confidences[i]
            color = colors[class_ids[i]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 20), (x + w, y), color, -1)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 5), font, 0.6, (255, 255, 255), 2)

    # Calculate FPS
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40), font, 0.8, (0, 0, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv3-Tiny Webcam", frame)

    # Press ESC to exit
    key = cv2.waitKey(1)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
