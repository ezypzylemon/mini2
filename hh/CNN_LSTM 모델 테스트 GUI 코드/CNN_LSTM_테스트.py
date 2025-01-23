import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# YOLO 로그 출력을 비활성화
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)


# 모델 로드
model_path = r"C:/Users/Admin/CNN_LSTM모델/mobilenet_lstm_model.h5"  # 저장된 모델 경로
model = tf.keras.models.load_model(model_path)
print(f"모델이 {model_path}에서 성공적으로 로드되었습니다.")

# YOLO 모델 로드
yolo_model = YOLO("C:/path/to/yolov8n-seg.pt")  # YOLO 경로 수정 필요

# 입력 크기 및 라벨 설정
sequence_length = 120  # 시퀀스 길이
image_size = (64, 64)  # 학습된 입력 크기
poses = ['squat', 'side', 'back']  # 분류 라벨

# 데이터 수집 변수 초기화
sequence = []  # 입력 시퀀스 저장
collecting = False  # 데이터 수집 상태
fps = 30  # 프레임 속도
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, fps)

if not cap.isOpened():
    print("Camera could not be opened. Check your connection.")
    exit()

print("Press SPACEBAR to start/stop data collection. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Invalid frame. Exiting.")
        break

    # YOLO를 사용하여 사람 객체 추출
    results = yolo_model(frame)
    person_frame = np.zeros_like(frame)

    # 마스크 적용 (사람만 흰색으로 처리)
    for idx, mask in enumerate(results[0].masks.data):
        if int(results[0].boxes.cls[idx]) == 0:  # Class ID 0 = person
            mask = np.array(mask.cpu().numpy(), dtype=np.uint8) * 255
            for c in range(3):
                person_frame[:, :, c] = np.where(mask > 0, 255, person_frame[:, :, c])

    # 데이터 수집 시작/중지
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # 스페이스바로 데이터 수집 제어
        collecting = not collecting
        if collecting:
            print("Data collection started.")
            sequence = []  # 기존 시퀀스 초기화
        else:
            print("Data collection stopped.")
            # 수집 종료 시 모델에 데이터 적합
            if len(sequence) == sequence_length:
                input_sequence = np.expand_dims(sequence, axis=0)
                prediction = model.predict(input_sequence, verbose=0)
                probabilities = prediction[0]
                print("\n--- Prediction Probabilities ---")
                for pose, prob in zip(poses, probabilities):
                    print(f"{pose}: {prob * 100:.2f}%")
                predicted_pose = poses[np.argmax(probabilities)]
                print(f"Predicted Pose: {predicted_pose}\n")
            else:
                print(f"Not enough data collected. Required: {sequence_length}, Collected: {len(sequence)}")

    # 데이터 수집 중이라면 시퀀스에 추가
    if collecting:
        resized_frame = cv2.resize(person_frame, image_size) / 255.0  # 크기 조정 및 정규화
        sequence.append(resized_frame)
        if len(sequence) > sequence_length:  # 초과 데이터 방지
            sequence.pop(0)

        # 데이터 수집 상태 표시
        cv2.putText(person_frame, f"Collecting: {len(sequence)}/{sequence_length}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 결과 화면 출력
    cv2.imshow("Processed Frame", person_frame)
    cv2.imshow("Original Frame", frame)

    # ESC를 누르면 종료
    if key == 27:  # ESC key
        print("Exiting.")
        break

# 자원 정리
cap.release()
cv2.destroyAllWindows()
