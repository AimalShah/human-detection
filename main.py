#!/usr/bin/env python3
"""
Real-time object detection using a YOLO model and a local webcam (Raspberry Pi OS optimized).
This version uses the Picamera2 library for high FPS performance on Raspberry Pi.
OPTIMIZED FOR 20-30 FPS with better detection accuracy.
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
        default="/home/pi/ai/best2.pt",
        help="Path to YOLO .pt model weights. Use a 'nano' (yolov8n.pt) model for best speed.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=416,
        help="Capture width (lower = faster processing). 416 is optimal for YOLO.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=416,
        help="Capture height (lower = faster processing). 416 is optimal for YOLO.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every Nth frame (1=all, 2=every other). Default 1 for best detection.",
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
        
        # Warm up the model with a dummy inference
        dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=416, verbose=False, half=True)
        
        print("✅ Model loaded and warmed up")
    except Exception as e:
        print(f"❌ Error loading model '{args.model}': {e}")
        return

    # Picamera2 setup
    picam2 = None
    try:
        picam2 = Picamera2()
        
        # Configure for maximum speed with lower resolution
        config = picam2.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        picam2.configure(config)
        
        # Set additional controls for faster operation
        picam2.set_controls({
            "ExposureTime": 10000,  # Faster exposure (adjust if too dark)
            "AnalogueGain": 2.0      # Compensate for faster exposure
        })
        
        picam2.start()
        print(f"📸 Camera initialized at {args.width}x{args.height} @ 30 FPS")
        
        # Let camera warm up
        time.sleep(0.5)
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

    prev_time = 0
    show_fps = True
    frame_count = 0
    class_names = model.names

    print(f"⚡ Processing every {args.skip_frames} frame(s)")
    print("Controls: Q=quit, F=toggle FPS, S=start recording")

    # Pre-allocate frame for speed
    last_annotated_frame = None

    try:
        while True:
            loop_start = time.time()
            
            # Capture frame (fast operation)
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Process frame with YOLO
            if frame_count % args.skip_frames == 0:
                try:
                    # Run YOLO inference with optimized settings
                    results = model.predict(
                        frame,
                        imgsz=416,  # Fixed size for speed
                        verbose=False,
                        half=True,  # FP16 for speed
                        conf=0.35,  # Slightly lower confidence for better detection
                        iou=0.5,    # IoU threshold for NMS
                        device='cpu',
                        max_det=15,  # Allow more detections
                        agnostic_nms=True  # Faster NMS
                    )

                    # Draw detections
                    for result in results:
                        if hasattr(result, "boxes") and result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                label = f"{class_names.get(cls, str(cls))}: {conf:.2f}"

                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Draw label background and text
                                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), (0, 255, 0), -1)
                                cv2.putText(
                                    frame,
                                    label,
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    1,
                                )
                    
                    last_annotated_frame = frame.copy()
                    
                except Exception as e:
                    print(f"⚠️  Inference error: {e}")
            else:
                # Use last annotated frame if we're skipping
                if last_annotated_frame is not None:
                    frame = last_annotated_frame.copy()

            # FPS calculation
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
