import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow('IFSP', cv2.WINDOW_NORMAL)
cv2.resizeWindow('IFSP', 640, 480)

x, y = 100, 300
vx, vy = 5, 5

while True:
    _, frame1 = cap.read()
    # for i in range(5):
    #     _, _ = cap.read()
    _, frame2 = cap.read()
    frame = cv2.absdiff(frame1, frame2)
    print(f'frame.shape: {frame.shape}, frame.dtype: {frame.dtype}')
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.blur(frame, (5, 5), 0)

    # if x > 639 - 50 or x < 1 + 50:
    #     vx = -vx
    # if y > 479 - 50 or y < 1 + 50:
    #     vy = -vy
    # x = x + vx
    # y = y + vy
    # frame = cv2.circle(frame, 
    #                    (x, y), 
    #                    50, 
    #                    (0, 255, 0), 
    #                    thickness=3)

    cv2.imshow('IFSP', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Fim da captura")
cap.release()
cv2.destroyAllWindows()
