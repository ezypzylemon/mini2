import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
import json

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 데이터 저장 경로 및 동작 정의
DATA_PATH = 'C:/Users/Admin/Desktop/motion_data'
os.makedirs(DATA_PATH, exist_ok=True)

actions = ['어깨운동']
num_samples = 100

def collect_data(action):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라 초기화 실패. 연결된 카메라를 확인하세요.")
        return

    collected_data = []
    print(f"'{action}' 데이터를 수집합니다. 준비하세요!")

    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

    with mp_pose.Pose() as pose:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("프레임 캡처 실패. 계속 시도합니다...")
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                keypoints = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                collected_data.append(keypoints)

                # 관절만 그려진 이미지를 위한 빈 이미지 생성
                blank_image = np.zeros_like(frame)

                # 관절만 그리기
                mp_drawing.draw_landmarks(
                    blank_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # 원본 프레임에도 관절 그리기
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

            cv2.putText(frame, f"{action} 데이터 수집 중: {i+1}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1) & 0xFF  # key 변수를 루프 내에서 초기화

            if key == ord('a') and results.pose_landmarks:
                # 'a'를 누르면 관절만 그려진 이미지를 저장
                img_file_name = f"{action}_joints_{i+1:03d}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                img_file_path = os.path.join(action_path, img_file_name)
                print(f"이미지 저장 경로: {img_file_path}")  # 경로 확인을 위한 디버깅 메시지
                print(f"이미지 크기: {blank_image.shape}")  # 이미지 크기 확인
                if cv2.imwrite(img_file_path, blank_image):
                    print(f"관절 이미지 저장 성공: {img_file_path}")
                else:
                    print(f"관절 이미지 저장 실패: {img_file_path}. 저장 경로와 디스크 공간을 확인하세요.")
                i += 1

            if key == 27:  # esc 키를 누르면 데이터 수집 종료
                print("데이터 수집이 사용자의 요청으로 중단되었습니다.")
                break

    # 데이터 저장 완료 후에도 창을 띄워두기
    while True:
        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # esc 키를 누르면 창 닫기
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected_data:
        json_file_name = f"{action}_keypoints.json"
        json_file_path = os.path.join(DATA_PATH, json_file_name)
        with open(json_file_path, 'w') as json_file:
            json.dump(collected_data, json_file)
        print(f"'{json_file_name}' 파일 저장 완료!")

collect_data('어깨운동')
