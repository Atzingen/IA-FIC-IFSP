import cv2
import numpy as np

def emtpy_callback(_):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('IFSP')
# cv2.resizeWindow('Filtros', 640, 480)
screen_width, screen_height = 1920, 1080
padding_height = 100


cv2.createTrackbar("LH", "IFSP", 0, 179, emtpy_callback)
cv2.createTrackbar("LS", "IFSP", 0, 255, emtpy_callback)
cv2.createTrackbar("LV", "IFSP", 0, 255, emtpy_callback)
cv2.createTrackbar("UH", "IFSP", 179, 179, emtpy_callback)
cv2.createTrackbar("US", "IFSP", 255, 255, emtpy_callback)
cv2.createTrackbar("UV", "IFSP", 255, 255, emtpy_callback)
# cv2.setTrackbarPos("LH", "IFSP", 102)
# cv2.setTrackbarPos("LS", "IFSP", 122)
# cv2.setTrackbarPos("LV", "IFSP", 44)
# cv2.setTrackbarPos("UH", "IFSP", 123)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1000
params.maxArea = 1000000
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lh = cv2.getTrackbarPos("LH", "IFSP")
    ls = cv2.getTrackbarPos("LS", "IFSP")
    lv = cv2.getTrackbarPos("LV", "IFSP")
    uh = cv2.getTrackbarPos("UH", "IFSP")
    us = cv2.getTrackbarPos("US", "IFSP")
    uv = cv2.getTrackbarPos("UV", "IFSP")

    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])
    mask = cv2.inRange(hsv, lower, upper)

    inverted_mask = cv2.bitwise_not(mask)
    keypoints = detector.detect(inverted_mask)
    frame_with_keypoints = frame.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)
        frame_with_keypoints = cv2.circle(frame, (x, y), size, (0, 255, 255), thickness=5)


    frame_height, frame_width = frame.shape[:2]
    combined_width = 2 * frame_width
    scale_factor = min(screen_width / combined_width, screen_height / frame_height, 1)

    # Resize se necessário
    if scale_factor < 1:
        new_width = int(frame_width * scale_factor)
        new_height = int(frame_height * scale_factor)
        inverted_mask = cv2.resize(inverted_mask, (new_width, new_height))
        frame_with_keypoints = cv2.resize(frame_with_keypoints, (new_width, new_height))
    mask_bgr = cv2.bitwise_not(cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR))
    combined_frame = cv2.hconcat([mask_bgr, frame_with_keypoints])
    padded_frame = np.zeros((combined_frame.shape[0] + padding_height, combined_frame.shape[1], 3), dtype=np.uint8)
    padded_frame[:combined_frame.shape[0], :] = combined_frame

    cv2.imshow("IFSP", padded_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()