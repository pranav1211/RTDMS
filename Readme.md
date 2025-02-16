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
- Download the folder [Local Code](https://github.com/pranav1211/RTDMS/tree/main/Local%20Code)
- Run the file "app.py"
- Click the "Enable Audio" Button to start detection at the bottom of the screen

## How to Run Online
- Open the site : [RTDMS](https://Beyondmebtw.com/rtdms)
- Allow camera access permission
- Click the "Enable Audio" Button to start detection at the below dashboard
- Increase your phone volume
- A stable network connection is reccomended for better stability

## Results
- The application alerts users based on the following observations :
  1. **Driver Drowsy** : Eyes closed or squinting 
  2. **Head Up** : Head tilted up for more than 2 seconds
  3. **Head Down** : Head tilted down for more than 2 seconds
  4. **Yawning** : Mouth open 2 seconds
  5. **Phone In Use** : Phone detected in frame

- Images :
  - <img src="https://beyondmebtw.com/rtdms/headup.jpeg" alt="Head Up" width=75%>
  - <img src="https://beyondmebtw.com/rtdms/headdown.jpeg" alt="Head Down" width=75%>
  - <img src="https://beyondmebtw.com/rtdms/drowsy.jpeg" alt="Drowsy" width=75%>
  - <img src="https://beyondmebtw.com/rtdms/phone.jpeg" alt="Phone" width=75%>

## Credits

- This project, **Real-Time Driver Monitoring System**, was developed as part of the **Mini Project Work** course (Course Code: 24AM5PWMPW) during the 5th semester of the **Artificial Intelligence and Machine Learning** program at **BMS College of Engineering**.
- Team Members:
  - Pranav Veeraghanta (1BM22AI092)
  - Shreyas Sachin Kshatriya (1BM22AI125)
  - Mitesh J Upadhya (1BM22AI075)
  - Shrujal Srinath (1BM22AI127)

## **Project Insights**
For details about the system's architecture, literature review, and methodologies, kindly refer to the [Project Report](https://Beyondmebtw.com/rtdms/report.pdf).

## Contact us
If you would like to contribute, report issues, or have any questions about the project, feel free to reach out to us at [pranavv.ai22@bmsce.ac.in](mailto:pranavv.ai22@bmsce.ac.in), [shreyassk.ai22@bmsce.ac.in](mailto:shreyassk.ai22@bmsce.ac.in), [mitesh.ai22@bmsce.ac.in](mailto:mitesh.ai22@bmsce.ac.in). Contributions are always welcome!


## Licensing

This project is licensed under the **MIT License**:
- **[MIT License](LICENSE)**: A permissive license allowing free use, modification, and distribution for both personal and commercial purposes, with attribution.

For more details, see the [MIT License](LICENSE) file included in this repository.
