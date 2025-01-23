import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# GPU 메모리 증가 허용
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Mixed Precision 활성화
set_global_policy('mixed_float16')

# 데이터 경로 및 변수 설정
base_path = r"C:\Users\Admin\흑백YOLO"
poses = ['squat', 'side', 'back']
num_sequences = 100
sequence_length = 120
image_size = (64, 64)  # 이미지 크기 축소

# 데이터 로드 함수
def load_data():
    X, y = [], []
    for pose_idx, pose in enumerate(poses):
        for i in range(1, num_sequences + 1):
            folder_path = os.path.join(base_path, pose, str(i))
            sequence = []
            for frame in range(1, sequence_length + 1):
                image_path = os.path.join(folder_path, f"frame_{frame:03d}.jpg")
                img = load_img(image_path, target_size=image_size, color_mode='rgb')  # RGB로 로드
                img_array = img_to_array(img) / 255.0  # 정규화
                sequence.append(img_array)
            X.append(sequence)
            y.append(pose_idx)
    return np.array(X), to_categorical(np.array(y), num_classes=len(poses))

# 데이터 로드 및 분리
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MobileNet 모델 구성
cnn_model = MobileNet(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
model = Sequential([
    TimeDistributed(cnn_model, input_shape=(sequence_length, 64, 64, 3)),
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(256, return_sequences=False),
    Dense(128, activation='relu'),
    Dense(len(poses), activation='softmax', dtype='float32')  # Mixed Precision에서는 float32로 출력
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 학습
model.fit(X_train, y_train, epochs=20, batch_size=2, validation_split=0.2)  # 배치 크기 2
 
# 성능 평가 함수
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1) 
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=poses))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=poses, yticklabels=poses)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True") 
    plt.show() 

evaluate_model(model, X_test, y_test, "MobileNet")

# 모델 저장
model.save(r"C:\Users\Admin\CNN_LSTM모델\mobilenet_lstm_model.h5")
print("모델이 C:\\Users\\Admin\\CNN_LSTM모델\\mobilenet_lstm_model.h5에 저장되었습니다.")

