import cv2
import numpy as np
import os
from ultralytics import YOLO
import time

# Base data storage path
BASE_PATH = 'C:/Users/Admin/Desktop/motion_data_skeleton/side'
os.makedirs(BASE_PATH, exist_ok=True)

num_samples = 120  # Maximum number of frames to collect per action
fps = 30  # Frames per second 

def get_next_directory(base_path):
    """Get the next numbered directory in the base path."""
    existing_dirs = [int(d) for d in os.listdir(base_path) if d.isdigit()]
    next_dir = str(max(existing_dirs) + 1) if existing_dirs else '1'
    next_path = os.path.join(base_path, next_dir)
    os.makedirs(next_path, exist_ok=True)
    return next_path

def collect_skeleton_data(base_path, max_frames=120):
    cap = cv2.VideoCapture(0)  # Activate webcam
    cap.set(cv2.CAP_PROP_FPS, fps) 
    if not cap.isOpened():
        print("Camera could not be opened. Check your connection.")
        return

    print("Press SPACEBAR to start collecting data, or ESC to exit.")

    # Initialize YOLO Segmentation model
    model = YOLO("yolov8n-seg.pt")  # Load a pretrained YOLOv8 segmentation model

    while True:
        # Get the next directory for saving data
        save_path = get_next_directory(base_path)
        frame_count = 0
        collecting = False

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Invalid frame. Skipping this frame.")
                continue

            key = cv2.waitKey(1) & 0xFF

            # Show live feed before data collection starts
            if not collecting:
                cv2.putText(frame, "Press SPACE to start or ESC to exit", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Camera Preview', frame)

            # Start collecting on SPACEBAR
            if key == ord(' '):
                collecting = True
                print(f"Started collecting data. Saving to: {save_path}")
                start_time = time.time()

            # Collect frames until max_frames is reached
            if collecting:
                results = model(frame)
                person_frame = np.zeros_like(frame)

                # Extract person masks
                for idx, mask in enumerate(results[0].masks.data):
                    if int(results[0].boxes.cls[idx]) == 0:  # Check if class ID is 'person'
                        mask = np.array(mask.cpu().numpy(), dtype=np.uint8) * 255
                        for c in range(3):
                            person_frame[:, :, c] = np.where(mask > 0, 255, person_frame[:, :, c])

                # Save the current frame as an image
                img_file_name = f"frame_{frame_count + 1:03d}.jpg"
                img_file_path = os.path.join(save_path, img_file_name)

                if cv2.imwrite(img_file_path, person_frame):
                    print(f"Image saved: {img_file_path}")

                '''
                # Display the processed frame
                cv2.putText(person_frame, f"Collecting: {frame_count + 1}/{max_frames}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Person Data Collection', person_frame)
                ''' 
                
                # Display the processed frame with current session and frame number
                cv2.putText(person_frame, f"Session: {os.path.basename(save_path)} - Frame: {frame_count + 1}/{max_frames}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # cv2.putText(person_frame, f"Collecting: {frame_count + 1}/{max_frames}",
                #             (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Person Data Collection', person_frame)

 
                frame_count += 1

                # Check if we reached max_frames
                if frame_count >= max_frames:
                    print(f"Completed collecting {max_frames} frames. Data saved to: {save_path}")
                    collecting = False  # Stop collecting
                    frame_count = 0  # Reset frame count
                    break

                # Control frame rate
                elapsed_time = time.time() - start_time
                if elapsed_time < 1.0 / fps:
                    time.sleep(1.0 / fps - elapsed_time)
                    start_time = time.time()

            # Exit on ESC
            if key == 27:  # ESC key
                print("Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                return

            # **Ensure normal camera view after collection ends**
            if not collecting and frame_count == 0:
                cv2.putText(frame, "Press SPACE to start another collection or ESC to exit", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Camera Preview', frame)

                # Exit preview if ESC is pressed
                if key == 27:  # ESC key
                    print("Exiting.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

# Start collecting skeleton data
collect_skeleton_data(BASE_PATH, num_samples)
