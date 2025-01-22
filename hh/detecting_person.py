import cv2
import numpy as np
import os
from ultralytics import YOLO

# Data storage path and action definitions
DATA_PATH = 'C:/Users/Admin/Desktop/motion_data_skeleton'  # Change to a path with sufficient storage
os.makedirs(DATA_PATH, exist_ok=True)

actions = ['pose1', 'pose2', 'pose3']  # Define actions
num_samples = 100  # Number of samples to collect per action

def collect_skeleton_data(action, samples=100):
    cap = cv2.VideoCapture(0)  # Activate webcam
    print(f"Collecting data for '{action}'. Get ready!")

    # Create action-specific directory
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

    # Initialize YOLO Segmentation model
    model = YOLO("yolov8n-seg.pt")  # Load a pretrained YOLOv8 segmentation model

    for i in range(samples):
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Invalid frame. Skipping this frame.")
            continue

        # Perform object detection with segmentation
        results = model(frame)
        person_frame = np.zeros_like(frame)

        '''
        # Iterate through detections and extract person masks
        for mask in results[0].masks.data:  # Access segmentation masks
            mask = np.array(mask.cpu().numpy(), dtype=np.uint8) * 255
            for c in range(3):
                person_frame[:, :, c] = np.where(mask > 0, 255, person_frame[:, :, c])
        '''
        
        # Iterate through detections and extract person masks
        for idx, mask in enumerate(results[0].masks.data):  # Access segmentation masks
            if int(results[0].boxes.cls[idx]) == 0:  # Check if class ID is 'person'
                mask = np.array(mask.cpu().numpy(), dtype=np.uint8) * 255
                for c in range(3):
                    person_frame[:, :, c] = np.where(mask > 0, 255, person_frame[:, :, c])


        # Save the current frame as an image
        img_file_name = f"{action}_{i+1:03d}.jpg"
        img_file_path = os.path.join(action_path, img_file_name)

        # Check save result
        if cv2.imwrite(img_file_path, person_frame):
            print(f"Image saved: {img_file_path}")
        else:
            print(f"Failed to save image: {img_file_path}, Frame shape: {person_frame.shape}")

        # Display the frame
        cv2.putText(person_frame, f"Collecting {action}: {i+1}/{samples}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Person Data Collection', person_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Collect skeleton data for each action
for action in actions:
    collect_skeleton_data(action, num_samples)
