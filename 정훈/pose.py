import cv2
import numpy as np
import mediapipe as mp
import os
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Data storage path
BASE_PATH = 'C:/Users/Admin/Desktop/motion_data'
IMAGE_FOLDER_NAME = "images"
COORD_FOLDER_NAME = "좌표"
os.makedirs(BASE_PATH, exist_ok=True)

# Function to create a new numbered folder for images
def create_new_image_folder(base_path):
    folder_path = os.path.join(base_path, IMAGE_FOLDER_NAME)
    os.makedirs(folder_path, exist_ok=True)
    existing_folders = [int(name) for name in os.listdir(folder_path) if name.isdigit()]
    next_number = max(existing_folders) + 1 if existing_folders else 1
    sub_folder_path = os.path.join(folder_path, str(next_number))
    os.makedirs(sub_folder_path, exist_ok=True)
    return sub_folder_path, next_number

# Function to draw pose landmarks on a black background
def draw_landmarks_on_black(image, pose_landmarks):
    black_image = np.zeros_like(image)
    mp_drawing.draw_landmarks(
        black_image, pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )
    return black_image

# Main function to collect data
def collect_pose_data():
    cap = cv2.VideoCapture(0)
    is_saving = False
    collected_data = []
    image_folder_path, coord_folder_number = None, None
    frame_count = 0
    recording_cycles = 0  # 녹화 사이클 카운터
    waiting = False  # 대기 상태를 나타내는 변수
    cycle_start = False  # 사이클 시작 상태를 나타내는 변수

    print("Press 'SPACE' to start the recording cycle. Press 'ESC' to exit.")

    # Set the window size
    cv2.namedWindow("Pose Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Capture", 1280, 720)  # Set window size to 1280x720

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        # Convert frame to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw landmarks if available
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # If saving is active, store data
            if is_saving:
                keypoints = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                collected_data.append(keypoints)

                # Save image to the current folder
                img_file_name = f"frame_{frame_count:03d}.jpg"
                img_file_path = os.path.join(image_folder_path, img_file_name)
                black_background = draw_landmarks_on_black(frame, results.pose_landmarks)
                cv2.imwrite(img_file_path, black_background)
                frame_count += 1

                # Display the number of images saved
                cv2.putText(frame, f"Saved images: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red text

                # Check if 120 frames have been captured
                if frame_count >= 120:
                    is_saving = False
                    if collected_data:
                        npy_file_name = f"keypoints_{coord_folder_number}.npy"
                        npy_file_path = os.path.join(coord_folder_path, npy_file_name)
                        np.save(npy_file_path, np.array(collected_data))
                        print(f"Coordinates saved to: {npy_file_path}")
                    print("Stopped saving.")
                    recording_cycles += 1  # Increment cycle count
                    waiting = True  # Set waiting state

        # Show FPS in red text
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red text

        # Display "Recording" if saving
        if is_saving:
            cv2.putText(frame, "Recording", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green text

        # Display "Wait" if waiting
        if waiting:
            cv2.putText(frame, "Wait", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)  # Yellow text

        # Display "Cycle Start" if cycle is starting
        if cycle_start:
            cv2.putText(frame, "Cycle Start", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Blue text
            cycle_start = False  # Reset cycle start state

        # Display the frame
        cv2.imshow("Pose Capture", frame)

        # Handle key inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Spacebar pressed
            if recording_cycles == 0:  # Start only if not already started
                waiting = True  # Initial wait before starting
                cycle_start = True  # Set cycle start state
                print("Starting recording cycles...")

        elif key == 27:  # ESC pressed
            print("Exiting...")
            break

        # Start a new cycle if not saving and waiting is false
        if not is_saving and waiting and recording_cycles < 10:
            # Wait for 2 seconds before starting recording
            time.sleep(2)
            waiting = False  # Reset waiting state

            # Start a new cycle: create new folders
            image_folder_path, coord_folder_number = create_new_image_folder(BASE_PATH)
            coord_folder_path = os.path.join(BASE_PATH, COORD_FOLDER_NAME)
            os.makedirs(coord_folder_path, exist_ok=True)
            collected_data = []
            frame_count = 0  # Reset frame count
            print(f"Started saving to: {image_folder_path} and keypoints_{coord_folder_number}")
            is_saving = True

        # Stop after 10 cycles
        if recording_cycles >= 10:
            print("Completed 10 recording cycles.")
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

# Run the data collection
collect_pose_data()
