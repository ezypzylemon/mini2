import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 전체 관절 연결 정보 (MediaPipe 기반)
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # 머리와 상체
    (0, 4), (4, 5), (5, 6), (6, 8),  # 머리와 상체
    (9, 10), (11, 12),  # 어깨
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # 왼팔
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # 오른팔
    (11, 23), (12, 24), (23, 24),  # 엉덩이
    (23, 25), (25, 27), (27, 29), (27, 31),  # 왼다리
    (24, 26), (26, 28), (28, 30), (28, 32)  # 오른다리
]

# 랜드마크 색상 정의
landmark_color = 'red'
connection_color = 'blue'

# 애니메이션 속도 설정 (ms 단위)
ANIMATION_SPEED = 100

def load_npy_file(file_path):
    """
    .npy 파일에서 랜드마크 데이터를 로드합니다.
    Args:
        file_path (str): .npy 파일 경로.
    Returns:
        ndarray: (num_frames, num_joints, 3) 형태의 좌표 데이터.
    """
    try:
        data = np.load(file_path)
        # 단일 프레임 데이터를 다중 프레임 형식으로 변환
        if data.ndim == 2 and data.shape[1] == 3:
            data = np.expand_dims(data, axis=0)  # (1, num_joints, 3)

        # 데이터 유효성 검사
        if data.ndim != 3 or data.shape[2] != 3:
            raise ValueError(f"Invalid data shape: {data.shape}")

        # NaN 또는 inf 값 검사
        if not np.isfinite(data).all():
            raise ValueError("Data contains NaN or infinite values.")

        return data
    except Exception as e:
        raise ValueError(f"Failed to load .npy file: {e}")

def animate_pose(joint_data):
    """
    .npy 데이터를 애니메이션으로 시각화합니다.
    Args:
        joint_data (ndarray): (num_frames, num_joints, 3) 형태의 좌표 데이터.
    """
    num_frames = joint_data.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 초기 랜드마크와 연결선
    scatter = ax.scatter([], [], [], c=landmark_color, s=50, label="Landmarks")
    lines = [ax.plot([], [], [], color=connection_color, linewidth=2)[0] for _ in connections]

    def init():
        """초기화 함수."""
        scatter._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return [scatter, *lines]

    def update(frame):
        """프레임별 업데이트 함수."""
        landmarks = joint_data[frame]
        xs, ys, zs = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]

        # 랜드마크 업데이트
        scatter._offsets3d = (xs, ys, zs)

        # 연결선 업데이트
        for line, (start, end) in zip(lines, connections):
            line.set_data([landmarks[start, 0], landmarks[end, 0]], [landmarks[start, 1], landmarks[end, 1]])
            line.set_3d_properties([landmarks[start, 2], landmarks[end, 2]])

        ax.set_title(f"Frame {frame + 1}/{num_frames}")
        return [scatter, *lines]

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=ANIMATION_SPEED)

    # 3D 플롯 설정
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose Visualization')
    plt.legend()
    plt.show()

def open_file():
    """
    파일 대화상자를 열어 사용자가 .npy 파일을 선택하도록 함.
    """
    file_path = filedialog.askopenfilename(
        title="Select a .npy file",
        filetypes=(("NumPy Files", "*.npy"), ("All Files", "*.*"))
    )
    if file_path:
        try:
            joint_data = load_npy_file(file_path)
            log_label.config(text=f"[INFO] File loaded: {file_path}\nShape: {joint_data.shape}", fg="green")

            # 애니메이션 실행
            animate_pose(joint_data)
        except Exception as e:
            log_label.config(text=f"[ERROR] {e}", fg="red")

def on_esc_key(event):
    """
    Esc 키를 눌렀을 때 애플리케이션 종료.
    """
    print("[INFO] Application closed by user.")
    root.destroy()

# Tkinter GUI 구성
root = tk.Tk()
root.title("3D Pose Animator")
root.geometry("500x400")

# 파일 열기 버튼
open_button = tk.Button(root, text="Open .npy File", command=open_file, font=("Arial", 14), fg="white", bg="blue")
open_button.pack(pady=50)

log_label = tk.Label(root, text="Waiting for file...", font=("Arial", 12))
log_label.pack(pady=10)

# Esc 키 이벤트 연결
root.bind("<Escape>", on_esc_key)

# Tkinter 메인 루프 실행
root.mainloop()
