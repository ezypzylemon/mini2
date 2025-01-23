import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Data storage path and action definitions
DATA_PATH = 'C:/Users/Admin/Desktop/motion_data'  # Change to a path with sufficient storage
os.makedirs(DATA_PATH, exist_ok=True)

actions = ['pose1', 'pose2', 'pose3']  # Define actions
num_samples = 100  # Number of samples to collect per action

def collect_data(action, samples=100):
    cap = cv2.VideoCapture(0)  # Activate webcam
    collected_data = []

    print(f"Collecting data for '{action}'. Get ready!")

    # Create action-specific directory
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

    for i in range(samples):
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Invalid frame. Skipping this frame.")
            continue

        # Extract keypoints using MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Save keypoints only if available
        if results.pose_landmarks:
            keypoints = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
            collected_data.append(keypoints)

            # Draw pose landmarks on the current frame
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Save the current frame as an image
            img_file_name = f"{action}_{i+1:03d}.jpg"
            img_file_path = os.path.join(action_path, img_file_name)
            
            # Check save result
            if cv2.imwrite(img_file_path, frame):
                print(f"Image saved: {img_file_path}")
            else:
                print(f"Failed to save image: {img_file_path}, Frame shape: {frame.shape}")

        # Display the current frame
        cv2.putText(frame, f"Collecting {action}: {i+1}/{samples}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save keypoints as a .npy file
    if collected_data:
        npy_file_name = f"{action}_keypoints.npy"
        npy_file_path = os.path.join(DATA_PATH, npy_file_name)
        np.save(npy_file_path, np.array(collected_data))
        print(f"'{npy_file_name}' saved successfully!")


# Collect data for each action
for action in actions:
    collect_data(action, num_samples)

# Close the MediaPipe Pose object
pose.close()
