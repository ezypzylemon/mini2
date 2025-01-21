import cv2
import numpy as np
import threading
import time
import random
from playsound import playsound
from queue import Queue

# --------------------
# 글로벌 / 공유 변수
# --------------------
frame_queue = Queue(maxsize=2)  # 카메라 프레임을 저장할 큐
stop_flag = False               # 프로그램 전체 종료 여부 플래그
restart_flag = False            # 게임 재시작 여부 플래그

# 게임 상태
robot_status = 'blind'    # (blind, speaking, looking)
player_status = 'alive'   # (alive, dead)
MOVE_THRESHOLD = 500      # 움직임 임계값

# --------------------
# 1) 카메라 캡처 스레드
# --------------------
def camera_thread(index=1):
    global stop_flag
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 윈도우에서는 CAP_DSHOW 사용
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        stop_flag = True
        return

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽지 못했습니다.")
            time.sleep(0.1)
            continue
        
        # 큐가 꽉 차 있으면 오래된 프레임을 버리고 새 프레임을 넣어 최신 상태 유지
        if frame_queue.full():
            _ = frame_queue.get()
        frame_queue.put(frame)

    cap.release()
    print("카메라 스레드 종료")

# --------------------
# 2) 게임(사운드) 로직 스레드
# --------------------
def game_logic_thread():
    global stop_flag
    global robot_status, player_status

    while not stop_flag:
        # 로봇이 blind 상태일 때 -> speaking으로 전환 -> 사운드 재생 -> looking -> ...
        if robot_status == 'blind':
            robot_status = 'speaking'
            
            # 랜덤 사운드 재생 (블로킹)
            rand_sound = random.randint(1, 6)
            sound_path = "C:/Users/Admin/Desktop/삼정/squid/younghee/sound/squid_game_" + str(rand_sound) + ".mp3"  # 윈도우 경로 수정
            try:
                playsound(sound_path)
            except:
                pass  # 사운드 파일 미존재 등 예외 발생 시 무시
            
            # speaking 끝나면 -> looking
            robot_status = 'looking'
            time.sleep(3)
            
            # looking 끝났고, 아직 플레이어가 살아있다면 다시 blind로
            if player_status == 'alive':
                robot_status = 'blind'
        else:
            # 그 외에는 잠깐씩 대기
            time.sleep(0.1)

    print("게임 로직 스레드 종료")

# --------------------
# 3) 메인 스레드 (영상 표시 + 배경 차감)
# --------------------
def main_loop():
    global stop_flag
    global robot_status, player_status, restart_flag

    # 배경 차감 초기화
    sub = cv2.createBackgroundSubtractorKNN(history=1, dist2Threshold=500, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    while not stop_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            # 배경 차감으로 움직임 확인
            mask = sub.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            diff = cv2.countNonZero(mask)
            # looking 중에 움직임이 기준 이상이면 'dead' 처리
            if robot_status == 'looking' and diff > MOVE_THRESHOLD:
                player_status = 'dead'

            # 화면에 상태 표시
            cv2.putText(frame, f"Robot: {robot_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Player: {player_status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # 추가: dead 상태라면 큰 문구 표시
            if player_status == 'dead':
                cv2.putText(frame,
                            "YOU ARE DEAD!",
                            (100, 160),  # 출력 위치 (적절히 조정)
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,           # 폰트 크기
                            (0, 0, 255), # 빨간색
                            3)           # 두께

            cv2.imshow("Detection", frame)
            cv2.imshow("mask", mask)

        # 키 이벤트 처리: esc 키로 종료, R 키로 게임 재시작
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키
            stop_flag = True
        elif key == ord('r'):  # R 키
            restart_flag = True
            stop_flag = True  # 현재 게임을 종료한 후 재시작하도록 플래그 설정

    cv2.destroyAllWindows()
    print("메인 루프 종료")

# --------------------
# 4) 실행부
# --------------------
def restart_game():
    global stop_flag, restart_flag, robot_status, player_status
    # 게임을 리셋하는 함수
    stop_flag = False
    restart_flag = False
    robot_status = 'blind'
    player_status = 'alive'

    # 카메라 스레드 초기화
    cam_t = threading.Thread(target=camera_thread, args=(1,), daemon=True)
    cam_t.start()

    # 게임 스레드 초기화
    game_t = threading.Thread(target=game_logic_thread, daemon=True)
    game_t.start()

    # 메인 루프 실행
    while True:
        main_loop()
        if restart_flag:  # 재시작 플래그가 있으면 게임을 리셋
            print("게임을 재시작합니다.")
            stop_flag = False
            restart_flag = False
            robot_status = 'blind'
            player_status = 'alive'
            continue  # 게임 리셋 후 다시 main_loop()를 호출

    # 스레드 종료 대기
    cam_t.join()
    game_t.join()
    print("프로그램 완전히 종료")

# 프로그램 실행
if __name__ == "__main__":
    restart_game()  # 게임 시작
    # 게임이 종료되거나 재시작될 때까지 기다린 후 종료됩니다.
