import cv2
import numpy as np
import threading
import time
import random
import torch
from mediapipe import solutions as mp_solutions
from playsound import playsound
from queue import Queue

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------
# 글로벌 / 공유 변수
# --------------------
frame_queue = Queue(maxsize=20)  # 카메라 프레임을 저장할 큐 (크기 증가)
stop_flag = True               # 프로그램 전체 종료 여부 플래그
restart_flag = False           # 게임 재시작 여부 플래그
start_game_flag = False        # 게임 시작 여부 플래그

# 게임 상태
robot_status = 'blind'    # (blind, speaking, looking, mission)
player_status = 'alive'   # (alive, dead)
mission_class = None      # 부여된 미션 동작
mission_success = None    # 미션 성공 여부 초기화
MOVE_THRESHOLD = 1000     # 움직임 임계값 (더 높은 값으로 조정)
MOVE_DURATION = 1.0       # 움직임 감지 지연 시간 (초)

# LSTM 모델 초기화
model_path = "./lstm_model_scripted.pt"  # TorchScript 저장된 LSTM 모델
model = torch.jit.load(model_path).to(device)  # 모델을 GPU로 이동
model.eval()
labels_map = {0: "Back", 1: "Squat", 2: "Side"}  # LSTM 클래스 라벨 매핑

# Mediapipe 초기화
mp_pose = mp_solutions.pose
pose = mp_pose.Pose()

# 버퍼 크기 설정 (30프레임 = 1초)
buffer_limit = 30
keypoints_buffer = []

# 움직임 감지 시간 기록
move_detected_time = None

# 프레임 속도 측정
fps = 0
frame_count = 0
start_time = time.time()

# --------------------
# 1) 카메라 캡처 스레드
# --------------------
def camera_thread():
    global stop_flag, fps, frame_count, start_time

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 해상도 낮춤 (640x360)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 프레임 속도를 30FPS로 설정

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        stop_flag = True
        return

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # 프레임 속도 측정
        frame_count += 1
        if frame_count >= 30:
            fps = frame_count / (time.time() - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

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
                mission_success = None  # 초기화

        elif robot_status == 'mission':
            if mission_success is True:
                robot_status = 'blind'
            elif mission_success is False:
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

# --------------------
# 4) 메인 스레드
# --------------------
def main_loop():
    global stop_flag, robot_status, player_status, mission_class, mission_success, start_game_flag, restart_flag, keypoints_buffer, move_detected_time

    sub = cv2.createBackgroundSubtractorKNN(history=1, dist2Threshold=500, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while not stop_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # 프레임 크기를 전체화면으로 조정
            frame = cv2.resize(frame, (640, 360))

            mask = sub.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            diff = cv2.countNonZero(mask)
            if robot_status == 'looking' and diff > MOVE_THRESHOLD:
                if move_detected_time is None:
                    move_detected_time = time.time()  # 움직임 감지 시간 기록
                elif time.time() - move_detected_time >= MOVE_DURATION:
                    player_status = 'dead'  # 일정 시간 동안 움직임이 지속되면 죽음
            else:
                move_detected_time = None  # 움직임이 없으면 시간 기록 초기화

            if robot_status == 'mission':
                keypoints = extract_keypoints(frame)
                if keypoints is not None:
                    keypoints_buffer.append(keypoints)
                    print(f"Buffer size: {len(keypoints_buffer)}")  # 버퍼 크기 로그 출력
                    if len(keypoints_buffer) >= buffer_limit:
                        keypoints_array = np.array(keypoints_buffer)  # (buffer_limit, 33, 3)
                        keypoints_normalized = (keypoints_array - keypoints_array.mean()) / (keypoints_array.std() + 1e-8)
                        input_data = torch.tensor(keypoints_normalized).float().to(device)  # 데이터를 GPU로 이동

                        input_data = input_data.view(1, buffer_limit, -1)  # (1, buffer_limit, 33*3)

                        with torch.no_grad():
                            outputs = model(input_data)
                            predicted_class = torch.argmax(outputs, dim=1).item()
                            predicted_label = labels_map[predicted_class]
                            print(f"Predicted Label: {predicted_label}, Mission: {mission_class}")

                        if predicted_label == mission_class:
                            mission_success = True
                            # 성공 시 초록색 화면 표시
                            frame[:] = (0, 255, 0)  # 전체 화면을 초록색으로
                        else:
                            mission_success = False
                            # 실패 시 빨간색 화면 표시
                            frame[:] = (0, 0, 255)  # 전체 화면을 빨간색으로

                        keypoints_buffer = []  # 버퍼 초기화

            # 상태 정보 표시 (가독성 향상)
            text_background = np.zeros_like(frame, dtype=np.uint8)
            cv2.rectangle(text_background, (10, 10), (600, 150), (0, 0, 0), -1)  # 반투명 배경
            frame = cv2.addWeighted(frame, 1.0, text_background, 0.5, 0)

            cv2.putText(frame, f"Robot: {robot_status}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, f"Player: {player_status}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, f"Mission: {mission_class}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # 미디어파이프 관절 데이터 시각화
            if robot_status == 'mission':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(frame)
                if result.pose_landmarks:
                    mp_solutions.drawing_utils.draw_landmarks(
                        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 전체화면으로 출력
            cv2.namedWindow("Detection", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
    global stop_flag, restart_flag, robot_status, player_status, start_game_flag, keypoints_buffer, move_detected_time
    stop_flag = False
    restart_flag = False
    robot_status = 'blind'
    player_status = 'alive'
    start_game_flag = False
    keypoints_buffer = []  # 버퍼 초기화
    move_detected_time = None  # 움직임 감지 시간 초기화

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
            keypoints_buffer = []  # 버퍼 초기화
            move_detected_time = None  # 움직임 감지 시간 초기화
        else:
            break

    cam_t.join()
    game_t.join()
    print("프로그램 완전히 종료")

if __name__ == "__main__":
    restart_game()