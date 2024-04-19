import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture('pose2.mp4')

# Get video parameters
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare the video writer
output_video_path = r'C:\Users\user\Desktop\thirdeye\sameer project body\output\try2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Perform pose detection
        results = pose.process(image)

        # Create a black background image
        background = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Draw the pose annotations on the black image
        if results.pose_landmarks:
            # Normalize landmarks
            landmarks = results.pose_landmarks.landmark
            # Use the right hip as the center point
            hip_x = landmarks[24].x
            hip_y = landmarks[24].y
            for landmark in landmarks:
                landmark.x = (landmark.x - hip_x) + 0.5
                landmark.y = (landmark.y - hip_y) + 0.5
                landmark.z = 0  # Flatten to 2D since we are not using Z-axis here

            # Convert normalized landmarks to pixel coordinates
            landmark_px = [(int(lm.x * frame_width), int(lm.y * frame_height)) for lm in landmarks]

            # Draw lines between landmarks
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
                    start_point = landmark_px[start_idx]
                    end_point = landmark_px[end_idx]
                    cv2.line(background, start_point, end_point, (255, 255, 255), 2)

            # Draw landmarks
            for lm in landmark_px:
                cv2.circle(background, lm, 5, (255, 0, 0), -1)

        # Write the frame to the output video
        out.write(background)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
