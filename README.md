# Vision Embedded

Vision module developed for the 2025 Embedded Software Contest.

This project identifies registered users and recognizes their current posture from a camera stream.  
Face recognition and pose estimation results are combined and transmitted to other embedded devices through MQTT.

## Key Features
- Face recognition using InsightFace embeddings
- Pose estimation using YOLOv8 Pose
- Posture classification: Standing, Sitting, HandsUp, Falling/Lying
- MQTT-based event and device status transmission
- Real-time camera processing with OpenCV

## Tech Stack
Python · OpenCV · YOLOv8 · InsightFace · MQTT
