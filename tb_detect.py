import cv2
from ultralytics import YOLO
import time

# Load a lightweight model for realtime; switch to 'yolov8s.pt' for better accuracy
model = YOLO("yolo_tb_finetune.pt")

# 0 is default camera; use 1,2,... if you have multiple
cap = cv2.VideoCapture(0)
# Optional: set resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Inference (set stream=True to avoid copying results to disk)
    results = model.predict(frame, conf=0.25, verbose=False)

    # Draw detections on the frame (Ultralytics provides ready-made plotting)
    annotated = results[0].plot()  # returns a numpy array with boxes/labels drawn

    # FPS meter
    now = time.time()
    fps = 0.9*fps + 0.1*(1.0/(now - prev))
    prev = now
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("YOLOv8 + OpenCV (Webcam)", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
