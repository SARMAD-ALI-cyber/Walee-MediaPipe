import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

model_path = 'F:\\Mediapipe_dataset\\DetectionWork\\blaze_face_short_range.tflite'

mp_drawing = mp.solutions.drawing_utils
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the video mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=0.6)

cap = cv2.VideoCapture("F:\\Mediapipe_dataset\\orignalDataset\\Videos\\FullLengthVideos\\orignal1_output_video.avi")
fps = cap.get(cv2.CAP_PROP_FPS)

with FaceDetector.create_from_options(options) as detector:
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Calculate timestamp in milliseconds
        frame_timestamp_ms = int((frame_number / fps) * 1000)

        # Perform face detection
        face_detector_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        if face_detector_result.detections:
            for detection in face_detector_result.detections:
                # Extract bounding box information
                bbox = detection.bounding_box
                x_min, y_min = int(bbox.origin_x), int(bbox.origin_y)
                width, height = int(bbox.width), int(bbox.height)
                
                # Draw bounding box on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('MediaPipe Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

cap.release()
cv2.destroyAllWindows()
