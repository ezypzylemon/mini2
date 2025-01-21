import cv2

def capture_and_split():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        mid_x = width // 2

        # 화면 분할
        cv2.line(frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)
        cv2.imshow("Game Screen", frame)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
