import cv2
import mediapipe as mp
import numpy as np
import torch

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 모델 불러오기
model_path = "C:/Users/Admin/Desktop/model_scripted.pt"  # 사전 저장된 모델 경로
model = torch.jit.load(model_path)
model.eval()

# 카테고리와 라벨 매핑 (영어 이름으로 수정)
labels_map = {0: "Back", 1: "Squat", 2: "Side"}

# 필요한 Mediapipe 관절 ID (17개 주요 관절)
important_joint_indices = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
]

# 실시간 예측 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mediapipe로 관절 추출
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image)

    # 관절 데이터 추출
    if result.pose_landmarks:
        # 33개의 관절 중 필요한 17개만 선택
        keypoints = np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark])
        keypoints = keypoints[important_joint_indices]  # 17개 관절만 선택
        keypoints = keypoints[np.newaxis, :, :]  # 배치 추가 (1, 17, 2)

        # 데이터 정규화 (0~1)
        keypoints_min = keypoints.min(axis=(1, 2), keepdims=True)
        keypoints_max = keypoints.max(axis=(1, 2), keepdims=True)
        keypoints_normalized = (keypoints - keypoints_min) / (keypoints_max - keypoints_min + 1e-8)

        # 모델 입력 형태 변환 (배치, 시퀀스 길이, 관절 수 * 좌표 수)
        input_data = torch.tensor(keypoints_normalized).float()
        input_data = input_data.view(1, -1, keypoints_normalized.shape[1] * keypoints_normalized.shape[2])

        # 모델 예측
        with torch.no_grad():
            outputs = model(input_data)  # 모델 출력: logits
            probabilities = torch.softmax(outputs, dim=1)  # 확률로 변환
            predicted_class = torch.argmax(probabilities, dim=1).item()  # 예측 클래스
            confidence = probabilities[0, predicted_class].item()  # 예측 확률

        # 출력 결과
        predicted_label = labels_map[predicted_class]
        confidence_percent = confidence * 100  # 퍼센트로 변환
        print(f"Predicted: {predicted_label}, Confidence: {confidence_percent:.2f}%")

        # 화면에 결과 표시
        cv2.putText(frame, f"Action: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence_percent:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 영상 출력
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
        break

# 종료
cap.release()
cv2.destroyAllWindows()
