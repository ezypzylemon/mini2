from camera.camera_handler import capture_and_split
from game.tetris_engine import TetrisGame
from model.predict import classify_action

def main():
    print("게임을 시작합니다!")
    # 카메라 캡처 시작
    capture_and_split()

    # 테트리스 게임 실행
    game = TetrisGame()
    game.start()

if __name__ == "__main__":
    main()
