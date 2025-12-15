#!/usr/bin/env python3
"""
Real-time object detection using a YOLO model and a local webcam (Raspberry Pi OS optimized).
This version uses the Picamera2 library for high FPS performance on Raspberry Pi.
"""

import argparse
import time
import sys
import cv2
from ultralytics import YOLO

# --- ADDED IMPORTS FOR PICAMERA2 ---
from picamera2 import Picamera2
# We no longer need 'from libcamera import controls'
# as we removed the unsupported set_controls line
# -----------------------------------


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
    # Removed the unused --cam argument as Picamera2 finds the CSI camera
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width (lower = faster processing). Try 320 for maximum FPS.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height (lower = faster processing). Try 240 for maximum FPS.",
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
        return

    # --- PICAMERA2 SETUP (Replaces cv2.VideoCapture) ---
    picam2 = None
    try:
        picam2 = Picamera2()
        
        # 1. Configure for fast video capture
        # Use RGB888 format for fast camera transfer, then convert to BGR for OpenCV
        config = picam2.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"}
        )
        picam2.configure(config)

        # 2. Start the camera
        picam2.start()
        print(f"📸 Camera initialized at {args.width}x{args.height} using Picamera2.")
    except Exception as e:
        print(f"❌ Error initializing Picamera2: {e}")
        print("Please ensure your camera is connected and the python3-picamera2 package is installed correctly.")
        return
    # ----------------------------------------------------

    # Video Writer setup (if --save is used)
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Or 'M' 'J' 'P' 'G'
    
    # Optional: Initializing writer if --save is passed at startup
    if args.save:
        out_path = "output.avi"
        writer = cv2.VideoWriter(out_path, fourcc, 20.0, (args.width, args.height))
        print(f"🎥 Saving video to {out_path}")


    prev_time = 0
    show_fps = True
    
    # Get a list of the class names for labeling the boxes
    class_names = model.names

    try:
        while True:
            # --- FAST FRAME CAPTURE (Replaces cap.read()) ---
            # Capture frame as a NumPy array (RGB format)
            frame_rgb = picam2.capture_array()
            # Convert the array to OpenCV's required BGR format
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            # ----------------------------------------------

            # Run YOLO inference
            # Using half=True is CRITICAL for speeding up inference on the RPi CPU
            results = model.predict(
                frame, 
                imgsz=(args.width, args.height), # Ensure YOLO processes at capture size
                verbose=False, 
                half=True, # Optimized for speed
                conf=0.4 # Use a confidence threshold to filter out weak detections
            )

            # Process and draw results
            for result in results:
                # Iterate through all detected bounding boxes
                if hasattr(result, "boxes"):
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = f"{class_names.get(cls, str(cls))}: {conf:.2f}"
    
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label background and text
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0), # Text color black for contrast
                            2,
                        )

            # FPS calculation and display
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
        # --- CLEANUP (Replaces cap.release()) ---
        if picam2 is not None:
            picam2.stop()
        # ----------------------------------------
        
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ Cleaned up and exited successfully.")


if __name__ == "__main__":
    main()
