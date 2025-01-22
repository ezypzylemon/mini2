import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

class PoseDetector:
    def __init__(self):
        self.pose = mp_pose.Pose()

    def detect_pose(self, frame):
        results = self.pose.process(frame)
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            return landmarks
        return None
