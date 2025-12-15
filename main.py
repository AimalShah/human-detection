#!/usr/bin/env python3
"""
Real-time object detection using a YOLO model and a local webcam (Raspberry Pi OS optimized).
This version uses the Picamera2 library for high FPS performance on Raspberry Pi.
OPTIMIZED FOR 20-30 FPS with good detection accuracy.
"""

import argparse
import time
import sys
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO on a local webcam (default camera index 0)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/aimalshah/AI/human-detection/best.pt",
        help="Path to YOLO .pt model weights. Use a 'nano' (yolov8n.pt) model for best speed.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,  # Lower resolution for speed
        help="Capture width (lower = faster processing).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,  # Lower resolution for speed
        help="Capture height (lower = faster processing).",
    )
    parser.add_argument(
        "--inference-size",
        type=int,
        default=320,  # Match capture size
        help="YOLO inference size (lower = faster).",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=3,
        help="Process every Nth frame (default 3 for speed).",
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
        print("🔄 Loading YOLO model...")
        model = YOLO(args.model)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model '{args.model}': {e}")
        return

    # Picamera2 setup
    picam2 = None
    try:
        picam2 = Picamera2()
        
        # Configure for speed - lower resolution
        config = picam2.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        print(f"📸 Camera initialized at {args.width}x{args.height}")
        
        # Quick warmup
        time.sleep(0.3)
        for _ in range(5):
            _ = picam2.capture_array()
            
    except Exception as e:
        print(f"❌ Error initializing Picamera2: {e}")
        return

    # Video Writer setup
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    
    if args.save:
        out_path = "output.avi"
        writer = cv2.VideoWriter(out_path, fourcc, 20.0, (args.width, args.height))
        print(f"🎥 Saving video to {out_path}")

    prev_time = time.time()
    show_fps = True
    frame_count = 0
    class_names = model.names
    
    # Store last detection results
    last_boxes = []

    print(f"⚡ Processing every {args.skip_frames} frame(s) at {args.inference_size}x{args.inference_size}")
    print("Controls: Q=quit, F=toggle FPS, S=start recording")

    try:
        while True:
            # Capture frame
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Run inference only on every Nth frame
            if frame_count % args.skip_frames == 0:
                try:
                    # CRITICAL: Use smaller inference size and optimized settings
                    results = model.predict(
                        frame,
                        imgsz=args.inference_size,  # Small size for speed
                        verbose=False,
                        half=False,  # Disable FP16 if causing issues
                        conf=0.4,
                        iou=0.45,
                        device='cpu',
                        max_det=10,
                        agnostic_nms=True,
                        classes=None  # Detect all classes
                    )
                    
                    # Extract boxes for reuse
                    last_boxes = []
                    for result in results:
                        if hasattr(result, "boxes") and result.boxes is not None:
                            for box in result.boxes:
                                last_boxes.append({
                                    'xyxy': box.xyxy[0].tolist(),
                                    'conf': float(box.conf[0]),
                                    'cls': int(box.cls[0])
                                })
                                
                except Exception as e:
                    print(f"⚠️  Inference error: {e}")

            # Draw boxes (fast operation, done every frame)
            for box_data in last_boxes:
                x1, y1, x2, y2 = map(int, box_data['xyxy'])
                conf = box_data['conf']
                cls = box_data['cls']
                label = f"{class_names.get(cls, str(cls))}: {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(
                    frame, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
                )

            # FPS calculation
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            prev_time = cur_time

            if show_fps:
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )

            # Display
            cv2.imshow("YOLO Webcam Detection", frame)

            # Save if recording
            if writer is not None:
                writer.write(frame)

            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                show_fps = not show_fps
            elif key == ord("s") and writer is None:
                out_path = "output.avi"
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, (args.width, args.height))
                print(f"🎥 Started saving to {out_path}")
                
            frame_count += 1
                
    except KeyboardInterrupt:
        print("⛔ Interrupted by user.")

    finally:
        print("🛑 Shutting down...")
        if picam2 is not None:
            picam2.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ Cleaned up and exited successfully.")


if __name__ == "__main__":
    main()
