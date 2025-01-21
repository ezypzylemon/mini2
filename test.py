import cv2

# 아이폰 카메라는 보통 index 0 또는 1로 인식됨
cap = cv2.VideoCapture(1)  # 또는 cv2.VideoCapture(1) 시도
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("iPhone Camera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
