#!/usr/bin/env python3
"""
Real-time object detection using a YOLO model and a local webcam (Raspberry Pi OS optimized).
This version uses the Picamera2 library for high FPS performance on Raspberry Pi.
OPTIMIZED FOR 20-30 FPS with frame skipping and threading.
"""

import argparse
import time
import sys
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import threading
from queue import Queue


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
        default=416,  # Reduced from 640 for faster processing
        help="Capture width (lower = faster processing). 416 is optimal for YOLO.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=416,  # Reduced from 480 for faster processing
        help="Capture height (lower = faster processing). 416 is optimal for YOLO.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=2,
        help="Process every Nth frame (1=all, 2=every other, 3=every third). Higher = faster FPS.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to disk as output.avi.",
    )
    return parser.parse_args()


class FrameProcessor:
    """Threaded frame processor to run YOLO inference in parallel"""
    def __init__(self, model, width, height):
        self.model = model
        self.width = width
        self.height = height
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.last_result = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _process_loop(self):
        while self.running:
            if not self.input_queue.empty():
                frame = self.input_queue.get()
                try:
                    # Run YOLO inference
                    results = self.model.predict(
                        frame,
                        imgsz=416,  # Fixed size for consistency
                        verbose=False,
                        half=True,
                        conf=0.4,
                        device='cpu',  # Explicit CPU usage
                        max_det=10  # Limit detections for speed
                    )
                    self.last_result = results
                    # Store in output queue (discard old if full)
                    if self.output_queue.full():
                        self.output_queue.get()
                    self.output_queue.put(results)
                except Exception as e:
                    print(f"⚠️  Inference error: {e}")
            else:
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
                
    def submit_frame(self, frame):
        # Non-blocking submission - discard if queue is full
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except:
                pass
        try:
            self.input_queue.put_nowait(frame)
        except:
            pass
            
    def get_latest_result(self):
        # Non-blocking get
        if not self.output_queue.empty():
            self.last_result = self.output_queue.get()
        return self.last_result


def draw_detections(frame, results, class_names):
    """Draw bounding boxes and labels on frame"""
    if results is None:
        return frame
        
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
    return frame


def main():
    args = parse_args()

    # Load YOLO model
    try:
        print("🔄 Loading YOLO model...")
        model = YOLO(args.model)
        model.fuse()  # Fuse model layers for faster inference
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model '{args.model}': {e}")
        return

    # Picamera2 setup
    picam2 = None
    try:
        picam2 = Picamera2()
        
        # Configure for maximum speed
        config = picam2.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            controls={"FrameRate": 30}  # Request 30 FPS from camera
        )
        picam2.configure(config)
        picam2.start()
        print(f"📸 Camera initialized at {args.width}x{args.height} @ 30 FPS")
        
        # Let camera warm up
        time.sleep(1)
    except Exception as e:
        print(f"❌ Error initializing Picamera2: {e}")
        return

    # Initialize threaded processor
    processor = FrameProcessor(model, args.width, args.height)
    processor.start()
    print("🚀 Started threaded inference processor")

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
    latest_results = None

    print(f"⚡ Processing every {args.skip_frames} frame(s) for optimal speed")
    print("Controls: Q=quit, F=toggle FPS, S=start recording")

    try:
        while True:
            # Capture frame
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Submit frame for processing only every Nth frame
            if frame_count % args.skip_frames == 0:
                processor.submit_frame(frame.copy())
            
            # Get latest detection results (non-blocking)
            latest_results = processor.get_latest_result()
            
            # Draw detections from latest results
            if latest_results is not None:
                frame = draw_detections(frame, latest_results, class_names)

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
        processor.stop()
        if picam2 is not None:
            picam2.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ Cleaned up and exited successfully.")


if __name__ == "__main__":
    main()
