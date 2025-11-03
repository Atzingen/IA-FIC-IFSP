import argparse
import os
import time

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 Inference: show boxes + class name + confidence")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("runs", "detect_med", "yolo11_digits", "weights", "best.pt"),
        help="Path to model weights (.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source: webcam index like '0' or path to image/video/directory"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default=None, help="Device: 0, 1, 'cpu' (default: auto)")
    parser.add_argument("--show-fps", action="store_true", help="Overlay FPS on the frame")
    return parser.parse_args()


def resolve_source(src_str: str):
    # Convert numeric string to int for webcam, else return path
    if src_str.isdigit():
        return int(src_str)
    return src_str


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Tip: after training, the best weights are usually at runs/detect/<exp>/weights/best.pt"
        )

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")

    print("============================================================")
    print("YOLOv11 Inference")
    print(f"Model : {args.model}")
    print(f"Source: {args.source}")
    print(f"Device: {device}")
    if torch.cuda.is_available() and device != "cpu":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"Conf  : {args.conf}")
    print(f"ImgSz : {args.imgsz}")
    print("============================================================")

    # Load model
    model = YOLO(args.model)

    # Prepare source
    source = resolve_source(args.source)

    # Predict with streaming for real-time display and uniform handling (images/videos/webcam)
    results_gen = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
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

        if args.show_fps:
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


if __name__ == "__main__":
    main()
