import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Open video file or capture device.
video_path = "F:\\Mediapipe_dataset\\orignalDataset\\Videos\\FullLengthVideos\\orignal2_output_video.avi" 
cap = cv2.VideoCapture(video_path)

# Initialize Face Detection
'''
For face detection, we have two model selection options: model_selection = 0 or 1 
0: Short-range model, which is optimized for close-range (short-range) subjects.(optimized for front camera of mobile)
1: Long-range model, which is optimized for subjects that are more than 2 meters away.(optimized for back camera of mobile)
'''
with mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        
        # Convert the BGR image to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect faces
        results = face_detection.process(frame_rgb)
        
        # Draw face detections of each face.
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
        
        # Display the frame
        cv2.imshow('MediaPipe Face Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
