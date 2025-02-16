# Real-Time Driver Monitoring System

The Real-Time Driver Monitoring System (RTDMS) is a project aimed at improving road safety by employing Artificial Intelligence, Machine Learning, and Computer Vision technologies. The system focuses on detecting driver fatigue and distractions in real time by analyzing visual and behavioral cues such as eye movement, head posture, facial expressions, and object detection.

## Features
1. **Real-Time Monitoring**: Detects driver fatigue and distractions using live video feeds.
2. **Audio-Visual Alerts**: Provides immediate feedback to the driver to prevent accidents.
3. **Environment Adaptability**: Robustness in well-lit and medium-lit lighting conditions.
4. **Mobile-Friendly**: Can be used via your smartphone.

## Installation

### 1. Software Requirements
1. **Operating System**:
   - Compatible with Linux, Windows.
   - Ubuntu 20.04+ is recommended for stability with Python, OpenCV, and TensorFlow.
2. **Python Version**:
   - Python 3.9.10 (used in development) or any Python 3.6+ version that supports the required libraries.
3. **Libraries and Frameworks**:
   - Install the following libraries via `pip`:
     ```bash
     pip install opencv-python dlib numpy mediapipe tensorflow flask
     ```
   - Ensure that `shape_predictor_68_face_landmarks.dat` (required by dlib) is present in the working directory.

### 2. Hardware Requirements

- **Camera**:
  - A webcam (built-in or external) capable of delivering at least 720p resolution at 30 FPS.
  - Higher resolution (1080p) is preferable for better accuracy in facial landmark and object detection.

- **Processor**:
  - Minimum: Intel i5 (6th generation or equivalent AMD processor).
  - Recommended: Intel i7 or better, or an equivalent AMD Ryzen processor for smooth real-time processing.

- **RAM**:
  - Minimum: 4GB (might work but may face delays).
  - Recommended: 8GB or more for handling simultaneous processes (e.g., running Flask, TensorFlow Lite, and OpenCV).

- **GPU (Optional but Beneficial)**:
  - If available, a dedicated GPU (like NVIDIA GeForce GTX 1050 or better) will accelerate TensorFlow Lite inference and OpenCV processing.
  - For environments without GPUs, the system can run on the CPU but may experience slightly higher latency.


## How To Run Locally
- First ensure you meet the Hardware and software requirements mentioned above
- Download the folder (Local Code)
- Run the file "app.py"
- On the site, Click the "Enable Audio" Button to start detection




## Licensing

This project is available under two licenses:
- **[GPL License](LICENSE)**: For open-source use with copyleft obligations.
- **[CC BY-NC License](LICENSE-CC-BY-NC.md)**: For personal and non-commercial use.
