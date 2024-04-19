import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp

def main():
    st.title("Human Pose Estimation")

    # MediaPipe setup
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # File uploader allows the user to upload videos
    uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_file is not None:
        # Save the uploaded video file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(tfile.name)

        # Get video parameters for background preparation
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Prepare output video file for download
        output_video_path = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Initialize MediaPipe Pose
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Check if the video has been opened successfully
            if cap.isOpened():
                st.text("Video successfully loaded!")
                # Read and display frame by frame
                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    # If there's a valid frame read
                    if not ret:
                        st.text("No more frames to display.")
                        break

                    # Convert the frame to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame)

                    # Create a black background image
                    background = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

                    # Draw the pose annotations on the black image
                    if results.pose_landmarks:
                        # Normalize landmarks
                        landmarks = results.pose_landmarks.landmark
                        hip_x = landmarks[24].x
                        hip_y = landmarks[24].y
                        for landmark in landmarks:
                            landmark.x = (landmark.x - hip_x) + 0.5
                            landmark.y = (landmark.y - hip_y) + 0.5
                            landmark.z = 0  # Flatten to 2D since we are not using Z-axis here

                        # Draw pose on the black background
                        mp_drawing.draw_landmarks(
                            background, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )

                    # Write to output video file
                    out.write(background)

                    # Display the processed frame with pose and the one on black background
                    stframe.image(background, use_column_width=True)
                    cv2.waitKey(10)

                # Release everything if job is finished
                cap.release()
                out.release()

                # Provide a download link to the output video
                with open(output_video_path, 'rb') as file:
                    btn = st.download_button(
                        label="Download Video",
                        data=file,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
            else:
                st.text("Failed to open the video.")

if __name__ == "__main__":
    main()
