# **AI-Driven Crime Monitoring & Surveillance System**🚨

This project is a real-time intelligent surveillance system designed to enhance public safety. By combining YOLOv4 for object detection and LSTM (Long Short-Term Memory) networks for action recognition, the system identifies criminal activities, weapons, and aggressive behavior in live video feeds.

## **Team Members**

1. Bushra Bilal Shaikh
2. Ayan Sajidkhan Pathan

## **🌟 Key Features**

Weapon Detection: Identifies firearms and cold weapons using a custom-trained YOLOv4 model.

Aggression Recognition: Uses MediaPipe pose estimation and LSTM to distinguish between neutral and violent physical movements.

Hand Gesture Analysis: Detects suspicious grasping or reaching motions using specialized LSTM layers.

Real-time Alerts: Trigger-based audio notifications (alert.wav) when a threat is detected.

## **📁 Project Structure**

realtime.py - The main execution script for live monitoring.

hands_lstm_realtime.py - Logic for hand-tracking and gesture recognition.

pose_data_generation.py - Script used to generate training data for violent vs. neutral poses.

yolov4.cfg & coco.names - Configuration files for the object detection layer.

alert.wav - Audio file for system warnings.

## **🛠️ Tech Stack**

Computer Vision: OpenCV, MediaPipe

Deep Learning: TensorFlow, Keras (for LSTM models)

Object Detection: YOLOv4

Language: Python

## **🚀 Getting Started**

1. Clone the Repository
   git clone https://github.com/Shaikh10Bushra/ai-crime-monitoring-system.git
   cd ai-crime-monitoring-system
   
2. Install Dependencies
   pip install -r requirements.txt
   
3. Download Model Weights (Required)
Due to GitHub's file size limits, please download the following files from ["C:\Users\shaik\Desktop\Crime_Detection\LSTM\yolov4.weights"] and place them in the root directory:
yolov4.weights
lstm-hand-grasping.h5

4. Run the System
   python realtime.py
   
## **👨‍💻 Developer**

Bushra Shaikh - Electronics and Computer Engineering Student [https://www.linkedin.com/in/bushra-shaikh-18b43a362] 
