import os
import time

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# Simple Parameters (edit these lines)
# ============================================================
# Default path to the best model (update if your run name differs)
MODEL = os.path.join("runs", "detect_med", "yolo11_digits", "weights", "best.pt")

# Source can be:
# - 0 (webcam), 1, 2 ...
# - "path/to/image.jpg" or "path/to/video.mp4"
# - "path/to/folder" (images)
SOURCE = 0

# Inference settings
CONF = 0.25
IMGSZ = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # auto GPU/CPU
SHOW_FPS = True


# ============================================================
# Sanity checks and info
# ============================================================
if not os.path.exists(MODEL):
    raise FileNotFoundError(
        f"Model not found: {MODEL}\n"
        "Tip: after training, the best weights are usually at runs/detect/<exp>/weights/best.pt"
    )

print("============================================================")
print("YOLOv11 Inference")
print(f"Model : {MODEL}")
print(f"Source: {SOURCE}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available() and DEVICE != "cpu":
    try:
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
print(f"Conf  : {CONF}")
print(f"ImgSz : {IMGSZ}")
print("============================================================")


# ============================================================
# Load and run inference (streaming)
# ============================================================
model = YOLO(MODEL)

# Convert numeric string to int for webcam if needed
if isinstance(SOURCE, str) and SOURCE.isdigit():
    source = int(SOURCE)
else:
    source = SOURCE

results_gen = model.predict(
    source=source,
    conf=CONF,
    imgsz=IMGSZ,
    device=DEVICE,
    stream=True,
    verbose=False,
    show=False,
)

window = "YOLOv11 Inference"
last_t = time.time()
fps = 0.0

for result in results_gen:
    # result.plot() returns an annotated frame with boxes + labels + confidences
    frame = result.plot()

    if SHOW_FPS:
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow(window, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' or ESC
        break

cv2.destroyAllWindows()
