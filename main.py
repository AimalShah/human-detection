#!/usr/bin/env python3
"""
Real-time object detection using a YOLO model and a local webcam.
Modified to use Picamera2 for native Raspberry Pi Camera support.
"""

import argparse
import time
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2 # NEW: Import Picamera2

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
    # The --cam argument is ignored by Picamera2 as it defaults to the CSI camera.
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
    # The original --cam argument is kept but ignored, as Picamera2 handles the CSI camera
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help=argparse.SUPPRESS, # Hide this argument, it's not used
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YOLO model
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"❌ Error loading model '{args.model}': {e}")
        print("Please check the path to your .pt weights.")
        sys.exit(1)

    # --- Picamera2 Setup (Replaces cv2.VideoCapture) ---
    try:
        picam2 = Picamera2()
        
        # Configure the camera with the desired resolution and an XRGB format
        # XRGB is an efficient format that converts easily to OpenCV BGR
        config = picam2.create_video_configuration(
            main={"size": (args.width, args.height), "format": "XRGB8888"}
        )
        picam2.configure(config)
        picam2.start()

        # Wait for auto-exposure to settle
        time.sleep(0.5)
        print(f"✅ Picamera2 started with resolution {args.width}x{args.height}.")

    except Exception as e:
        print(f"❌ Failed to initialize Picamera2: {e}")
        print("Please check if the camera module is connected correctly.")
        sys.exit(1)
    # ----------------------------------------------------

    # --- Video Writer Setup (OpenCV remains the same) ---
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = "output.avi"
        writer = cv2.VideoWriter(out_path, fourcc, 20.0, (args.width, args.height))
        print(f"🎥 Started saving to {out_path}")
    
    # --- Main Loop ---
    prev_time = 0
    show_fps = True
    
    print("Running detection. Press 'q' to quit, 'f' for FPS toggle, 's' to start/stop save.")

    try:
        while True:
            # 1. Capture the frame as a NumPy array (RGB format)
            frame_rgb = picam2.capture_array()
            
            # 2. Convert from RGB (Picamera2 default) to BGR (OpenCV default)
            # This is the standard conversion needed for cv2 functions.
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Perform detection on the frame
            results = model(frame, verbose=False)
            
            # Annotate the frame with bounding boxes
            annotated_frame = results[0].plot(
                boxes=True, conf=True, line_width=2, font_size=0.6
            )
            
            # Since .plot() is also BGR, we can use the result directly
            frame = annotated_frame 

            # --- FPS and Display Logic (Unchanged) ---
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
                # Re-initialize writer if saving is toggled after the script started
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, (args.width, args.height))
                print(f"🎥 Started saving to {out_path}")


    except KeyboardInterrupt:
        print("⛔ Interrupted by user.")

    finally:
        # --- Clean up Picamera2 ---
        picam2.stop()
        # --------------------------
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ Cleaned up and exited successfully.")


if __name__ == "__main__":
    main()
