#!/usr/bin/env python3
"""
Real-time object detection using a YOLO model on Raspberry Pi Camera.
Uses GStreamer backend (libcamerasrc) for stability on modern Pi OS.
"""

import argparse
import time
import sys
import cv2
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO on Raspberry Pi using GStreamer."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/aimalshah/AI/human-detection/best.pt",
        help="Path to YOLO .pt model weights.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width for processing.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height for processing.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to disk as output.avi.",
    )
    return parser.parse_args()

def open_camera(width, height):
    """
    Creates a GStreamer pipeline string for the Raspberry Pi Camera.
    This bypasses the need for the problematic libcamerify wrapper.
    """
    # The pipeline connects the camera source (libcamerasrc) -> 
    # sets resolution/framerate -> converts format -> sends to OpenCV (appsink).
    gst_str = (
        "libcamerasrc ! "
        f"video/x-raw, width={width}, height={height}, framerate=30/1 ! "
        "videoconvert ! "
        "appsink"
    )
    print(f"🔌 Attempting GStreamer pipeline:\n   {gst_str}")
    
    # Use cv2.CAP_GSTREAMER to tell OpenCV to interpret the pipeline string
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    return cap

def main():
    args = parse_args()

    # Load YOLO model
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"❌ Error loading model '{args.model}': {e}")
        sys.exit(1)

    print("✅ Model loaded. Initializing PiCamera via GStreamer...")

    # Initialize Camera using GStreamer
    cap = open_camera(args.width, args.height)

    if not cap.isOpened():
        print("❌ Error: Could not open camera via GStreamer.")
        print("   Make sure the following package is installed:")
        print("   sudo apt install gstreamer1.0-libcamera")
        sys.exit(1)

    print("✅ PiCam connected! Press 'q' to quit, 'f' to toggle FPS.")

    # Video Writer setup
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = "output.avi"
        writer = cv2.VideoWriter(out_path, fourcc, 20.0, (args.width, args.height))
        print(f"🎥 Saving video to {out_path}")

    prev_time = 0
    show_fps = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame.")
                break

            # YOLO inference
            results = model(frame)
            res = results[0]

            # Draw detections
            if hasattr(res, "boxes") and len(res.boxes) > 0:
                for box in res.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy.tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names.get(cls, str(cls)) if hasattr(model, "names") else str(cls)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            # FPS Calculation
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time) if prev_time > 0 else 0
            prev_time = cur_time

            if show_fps:
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("PiCam YOLO Detection", frame)

            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                show_fps = not show_fps

    except KeyboardInterrupt:
        print("⛔ Interrupted.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ Cleaned up and exited successfully.")

if __name__ == "__main__":
    main()
