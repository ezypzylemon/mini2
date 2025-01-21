import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES  # tkinter-dnd2 패키지 필요
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 관절 연결 정보 정의 (MediaPipe 포즈 기반)
connections = [
    (11, 13), (13, 15),  # 왼팔
    (12, 14), (14, 16),  # 오른팔
    (0, 11), (0, 12), (11, 12)  # 몸통
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
        if data.ndim != 3 or data.shape[2] != 3:
            raise ValueError(f"Invalid data shape: {data.shape}")
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

def on_file_drop(event):
    """
    드래그 앤 드롭으로 파일을 로드하여 애니메이션을 실행합니다.
    Args:
        event: TkinterDnD 이벤트 객체.
    """
    file_path = event.data.strip()  # 파일 경로
    if file_path.endswith(".npy"):
        try:
            joint_data = load_npy_file(file_path)
            log_label.config(text=f"[INFO] File loaded: {file_path}\nShape: {joint_data.shape}", fg="green")

            # 애니메이션 실행
            animate_pose(joint_data)
        except Exception as e:
            log_label.config(text=f"[ERROR] {e}", fg="red")
    else:
        log_label.config(text="[ERROR] Unsupported file type. Please drop a .npy file.", fg="red")

def on_esc_key(event):
    """
    Esc 키를 눌렀을 때 애플리케이션 종료.
    """
    print("[INFO] Application closed by user.")
    root.destroy()

# Tkinter GUI 구성
root = TkinterDnD.Tk()
root.title("3D Pose Animator")
root.geometry("500x400")

# 드래그 앤 드롭 이벤트 연결
label = tk.Label(root, text="Drag and drop a .npy file here to visualize", font=("Arial", 14), fg="blue")
label.pack(pady=50)

log_label = tk.Label(root, text="Waiting for file...", font=("Arial", 12))
log_label.pack(pady=10)

root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_file_drop)

# Esc 키 이벤트 연결
root.bind("<Escape>", on_esc_key)

# Tkinter 메인 루프 실행
root.mainloop()
