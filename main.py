#!/usr/bin/env python3
"""
Real-time object detection using a YOLO model on Raspberry Pi.
Optimized for use with 'libcamerify'.

Usage:
  libcamerify python3 main.py
"""

import argparse
import time
import sys
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO on a local webcam (default camera index 0)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/aimalshah/AI/human-detection/best.pt",
        help="Path to YOLO .pt model weights.",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Camera index (0 = default webcam).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Resize width for processing (higher = slower).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Resize height for processing.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to disk as output.avi.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YOLO model
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"❌ Error loading model '{args.model}': {e}")
        sys.exit(1)

    print(f"✅ Model loaded. Attempting to open camera index {args.cam}...")

    # --- KEY CHANGE: Force V4L2 Backend ---
    # We use cv2.CAP_V4L2 to ensure it plays nicely with libcamerify
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    
    # --- KEY CHANGE: Force MJPG Format ---
    # This prevents the 'select timeout' or empty frame errors by using compressed video
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # Set Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam index {args.cam}.")
        sys.exit(1)

    # Warmup: Read a dummy frame to ensure the stream is actually ready
    ret, _ = cap.read()
    if not ret:
        print("⚠️ Warning: Initial frame read failed. Retrying...")
        time.sleep(1)
        ret, _ = cap.read()
        if not ret:
            print("❌ Error: Camera opened but failed to return frames.")
            sys.exit(1)

    print("✅ Webcam connected. Press 'q' to quit, 's' to start saving video.")

    # Optional: video writer setup
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    if args.save:
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
                # Sometimes on Pi, a single dropped frame happens. Don't exit immediately.
                continue 

            # YOLO inference
            results = model(frame) # YOLO handles resizing internally, usually better than manual resize
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

            # FPS display
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

            # Display window
            cv2.imshow("YOLO Webcam Detection", frame)

            # Save frame if recording
            if writer is not None:
                writer.write(frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                show_fps = not show_fps
            elif key == ord("s") and writer is None:
                out_path = "output.avi"
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, (args.width, args.height))
                print(f"🎥 Started saving to {out_path}")

    except KeyboardInterrupt:
        print("⛔ Interrupted by user.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ Cleaned up and exited successfully.")


if __name__ == "__main__":
    main()
