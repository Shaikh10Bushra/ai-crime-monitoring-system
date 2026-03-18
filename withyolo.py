import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from collections import deque
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

# Ensure file existence
def check_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        exit()

# Load the trained LSTM model
lstm_model_path = "lstm-hand-grasping.h5"
check_file(lstm_model_path)
model = load_model(lstm_model_path)

# Load YOLO model for weapon detection
yolo_weights = "yolov4.weights"
yolo_config = "yolov4.cfg"
coco_names = "coco.names"

check_file(yolo_weights)
check_file(yolo_config)
check_file(coco_names)

net = cv2.dnn.readNet(yolo_weights, yolo_config)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()

try:
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except Exception as e:
    print(f"Error in YOLO layers: {e}")
    exit()

# Load class names
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# MediaPipe Pose setup
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Define a sliding window to store the last 20 frames of data
data_window = deque(maxlen=20)

# Real-time video capture
cap = cv2.VideoCapture(0)

def make_landmark_timestep(results):
    """Extracts pose landmarks and returns them in a flattened array."""
    if not results.pose_landmarks:
        return []
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm if len(c_lm) == 132 else []  # Ensure correct size

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    height, width, _ = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    if results.pose_landmarks:
        lm = make_landmark_timestep(results)
        if lm:
            data_window.append(lm)

        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        if len(data_window) == 20:
            input_data = np.array(data_window).reshape(1, 20, 132)
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            label = "Neutral" if predicted_class == 1 else "Violent" if predicted_class == 2 else "Unknown"
            cv2.putText(frame, f'Action: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in ["knife", "gun"]:
                center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f'Weapon Detected: {classes[class_id]}', (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Real-time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
