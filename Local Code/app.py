from flask import Flask, render_template, Response, jsonify
import cv2
import dlib
import numpy as np
import mediapipe as mp
import time
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for drowsiness and yawning detection
EAR_THRESHOLD = 0.19
MAR_THRESHOLD = 0.40
YAWN_FRAMES = 10

# Constants for head pose detection
PITCH_THRESHOLD = 10   # degrees
HEAD_POSE_TIMER = 2    # seconds

# Global variables to store the current status and timers
current_status = "Active"
head_pose_start_time = None
yawn_frame_count = 0

##########################################################
# Phone Detection using TFLite model
##########################################################

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

##########################################################
# Helper Functions
##########################################################

def compute_ear(eye_points):
    vertical_dist1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical_dist2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def compute_mar(mouth_points):
    vertical_dist1 = np.linalg.norm(mouth_points[13] - mouth_points[19])
    vertical_dist2 = np.linalg.norm(mouth_points[14] - mouth_points[18])
    horizontal_dist = np.linalg.norm(mouth_points[12] - mouth_points[16])
    mar = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return mar

##########################################################
# Video Feed Generator
##########################################################

def generate_frames():
    global current_status, head_pose_start_time, yawn_frame_count

    cap = cv2.VideoCapture(0)  # Use the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from the camera.")
            break

        # Convert frame to RGB for processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        img_h, img_w, _ = image.shape

        # -----------------------------
        # Drowsiness & Yawning Detection using dlib
        # -----------------------------
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Compute EAR for both eyes
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            ear_left = compute_ear(left_eye)
            ear_right = compute_ear(right_eye)
            ear = (ear_left + ear_right) / 2.0

            # Compute MAR for the mouth
            mouth = landmarks[48:68]
            mar = compute_mar(mouth)

            # Drowsiness detection logic
            if ear < EAR_THRESHOLD:
                current_status = "Driver Drowsy!"
            else:
                current_status = "Active"

            # Yawning detection logic
            if mar > MAR_THRESHOLD:
                yawn_frame_count += 1
                if yawn_frame_count > YAWN_FRAMES:
                    current_status = "Yawning!"
            else:
                yawn_frame_count = 0

            # Draw EAR and MAR values
            cv2.putText(image, f"EAR: {ear:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"MAR: {mar:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # -----------------------------
        # Head Pose Estimation using MediaPipe
        # -----------------------------
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_2d = []
                face_3d = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = img_w
                cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                       [0, focal_length, img_h / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                ret, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                if ret:
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    pitch_angle = angles[0] * 360  # approximate pitch angle in degrees

                    if pitch_angle > PITCH_THRESHOLD:
                        if head_pose_start_time is None:
                            head_pose_start_time = time.time()
                        elif time.time() - head_pose_start_time > HEAD_POSE_TIMER:
                            current_status = "Head Up!"
                    elif pitch_angle < -PITCH_THRESHOLD:
                        if head_pose_start_time is None:
                            head_pose_start_time = time.time()
                        elif time.time() - head_pose_start_time > HEAD_POSE_TIMER:
                            current_status = "Head Down!"
                    else:
                        head_pose_start_time = None
                        if current_status not in ["Driver Drowsy!", "Yawning!"]:
                            current_status = "Active"

        # -----------------------------
        # Phone Detection using TFLite
        # -----------------------------
        # Retrieve expected input size from the model details
        input_shape = input_details[0]['shape']  # e.g., [1, height, width, channels]
        expected_height = input_shape[1]
        expected_width = input_shape[2]

        # Resize the image to the expected size
        resized_image = cv2.resize(image_rgb, (expected_width, expected_height))

        # Prepare the input tensor for TFLite model
        input_tensor = np.expand_dims(resized_image, axis=0)
        input_tensor = input_tensor.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        # Retrieve output tensors
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num_detections = interpreter.get_tensor(output_details[3]['index'])
        phone_detected = False
        num = int(num_detections[0])
        for i in range(num):
            # In the COCO label map, "cell phone" is typically class 77.
            if scores[0][i] > 0.5 and int(classes[0][i]) == 76:
                phone_detected = True
                box = boxes[0][i]
                # Box is [ymin, xmin, ymax, xmax] in normalized coordinates
                y_min = int(box[0] * img_h)
                x_min = int(box[1] * img_w)
                y_max = int(box[2] * img_h)
                x_max = int(box[3] * img_w)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, f"Phone: {(scores[0][i] -0.20):.2f}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if phone_detected:
            current_status = "Phone in Use!"

        # -----------------------------
        # Encode and yield the frame
        # -----------------------------
        ret, buffer = cv2.imencode('.jpg', image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

##########################################################
# Flask Routes
##########################################################

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(status=current_status)

@app.route('/')
def index():
    return render_template('index.html')

##########################################################
# Run the Flask App
##########################################################

if __name__ == "__main__":
    app.run(debug=True)
