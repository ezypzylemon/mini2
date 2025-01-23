# squid_win_lstm.py: "무궁화꽃이 피었습니다" 게임에 LSTM 모델을 사용한 미션 수행 추가

import cv2
import numpy as np
import threading
import time
import random
import torch
from mediapipe import solutions as mp_solutions
from playsound import playsound
from queue import Queue

# --------------------
# 글로벌 / 공유 변수
# --------------------
frame_queue = Queue(maxsize=2)  # 카메라 프레임을 저장할 큐
stop_flag = True               # 프로그램 전체 종료 여부 플래그
restart_flag = False           # 게임 재시작 여부 플래그
start_game_flag = False        # 게임 시작 여부 플래그

# 게임 상태
robot_status = 'blind'    # (blind, speaking, looking, mission)
player_status = 'alive'   # (alive, dead)
mission_class = None      # 부여된 미션 동작
mission_success = False   # 미션 성공 여부
MOVE_THRESHOLD = 500      # 움직임 임계값

# LSTM 모델 초기화
model_path = "./lstm_model_scripted.pt"  # TorchScript 저장된 LSTM 모델
model = torch.jit.load(model_path)
model.eval()
labels_map = {0: "Back", 1: "Squat", 2: "Side"}  # LSTM 클래스 라벨 매핑

# Mediapipe 초기화
mp_pose = mp_solutions.pose
pose = mp_pose.Pose()

# --------------------
# 1) 카메라 캡처 스레드
# --------------------
def camera_thread():
    global stop_flag

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        stop_flag = True
        return

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        if frame_queue.full():
            _ = frame_queue.get()
        frame_queue.put(frame)

    cap.release()
    print("카메라 스레드 종료")

# --------------------
# 2) 게임 로직 스레드
# --------------------
def game_logic_thread():
    global stop_flag, robot_status, player_status, mission_class, mission_success, start_game_flag

    while not stop_flag:
        if not start_game_flag:
            time.sleep(0.1)
            continue

        if robot_status == 'blind':
            robot_status = 'speaking'
            
            rand_sound = random.randint(1, 6)
            sound_path = f"./sound/squid_game_{rand_sound}.mp3"
            try:
                playsound(sound_path)
            except:
                pass

            robot_status = 'looking'
            time.sleep(3)

            if player_status == 'alive':
                robot_status = 'mission'
                mission_class = random.choice(list(labels_map.values()))
                mission_success = False

        elif robot_status == 'mission':
            if mission_success:
                robot_status = 'blind'
            else:
                player_status = 'dead'

        else:
            time.sleep(0.1)

    print("게임 로직 스레드 종료")

# --------------------
# 3) 관절 데이터 추출 및 전처리
# --------------------
def extract_keypoints(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image)
    if result.pose_landmarks:
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
        return keypoints
    return None

def preprocess_keypoints(keypoints):
    keypoints = np.expand_dims(keypoints, axis=0)  # (1, 33, 3)
    keypoints_normalized = (keypoints - keypoints.mean()) / (keypoints.std() + 1e-8)
    input_data = torch.tensor(keypoints_normalized).float()
    input_data = input_data.view(1, -1, 33 * 3)
    return input_data

# --------------------
# 4) 메인 스레드
# --------------------
def main_loop():
    global stop_flag, robot_status, player_status, mission_class, mission_success, start_game_flag, restart_flag

    sub = cv2.createBackgroundSubtractorKNN(history=1, dist2Threshold=500, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while not stop_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()

            mask = sub.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            diff = cv2.countNonZero(mask)
            if robot_status == 'looking' and diff > MOVE_THRESHOLD:
                player_status = 'dead'

            if robot_status == 'mission':
                keypoints = extract_keypoints(frame)
                if keypoints is not None:
                    input_data = preprocess_keypoints(keypoints)
                    with torch.no_grad():
                        outputs = model(input_data)
                        predicted_class = torch.argmax(outputs, dim=1).item()
                        predicted_label = labels_map[predicted_class]

                    if predicted_label == mission_class:
                        mission_success = True
                    else:
                        mission_success = False

            cv2.putText(frame, f"Robot: {robot_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Player: {player_status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Mission: {mission_class}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Detection", frame)
            cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            stop_flag = True
        elif key == ord('r'):
            restart_flag = True
            stop_flag = True
        elif key == ord(' '):
            start_game_flag = not start_game_flag

    cv2.destroyAllWindows()
    print("메인 루프 종료")

# --------------------
# 5) 실행부
# --------------------
def restart_game():
    global stop_flag, restart_flag, robot_status, player_status, start_game_flag
    stop_flag = False
    restart_flag = False
    robot_status = 'blind'
    player_status = 'alive'
    start_game_flag = False

    cam_t = threading.Thread(target=camera_thread, daemon=True)
    cam_t.start()

    game_t = threading.Thread(target=game_logic_thread, daemon=True)
    game_t.start()

    while True:
        main_loop()
        if restart_flag:
            stop_flag = False
            restart_flag = False
            robot_status = 'blind'
            player_status = 'alive'
            start_game_flag = False
        else:
            break

    cam_t.join()
    game_t.join()
    print("프로그램 완전히 종료")

if __name__ == "__main__":
    restart_game()
