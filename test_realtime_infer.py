import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from inference_brushing import (
    load_scaler, build_model, predict_one, DEFAULT_FEATURE_COLS
)

# Paths from your training script output
MODEL_PATH = "toothy_mlp.pt"
SCALER_PATH = "toothy_scaler.json"

# Load scaler + model (make sure hidden sizes & dropout match training)
mean, scale = load_scaler(SCALER_PATH)
in_dim = len(DEFAULT_FEATURE_COLS)                # + extras if you used use_relative_diffs=True (then add 10)
model = build_model(MODEL_PATH, in_dim=in_dim, hidden=(64, 32), dropout=0.10, device="cpu")

# One sample (keys must match training)
row = {
    "tb_mid_x": 12.3, "tb_mid_y": -8.1,
    "hw_x": -30.0, "hw_y": -20.0,
    "hm_x": -10.0, "hm_y": -5.0,
    "ht_x": 5.0,   "ht_y": 2.0,
    "is_smiling": 0
}

# load YOLO
yolo = YOLO("yolo_tb_finetune2.pt")
yolo.to("cuda")

# load mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# load mediapipe face
mp_fd = mp.solutions.face_detection
mp_fm = mp.solutions.face_mesh

# load mediapipe drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# load moondream
moondream = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda", # "cuda" on Nvidia GPUs
)

def yolo_predict(frame, draw=False, yolo=yolo):
  yolo_res = yolo.predict(frame, conf=0.25, verbose=False, device="cuda:1")
  boxes = yolo_res[0].boxes
  
  # Filter boxes for class ID 14
  target_class = 14
  tb_bbox = None
    
  # get the highest confidence bounding box for the tooth brush
  for box in boxes:
    if int(box.cls[0]) == 14:
      x1, y1, x2, y2 = box.xyxy[0].tolist()  # bounding box coords
      conf = float(box.conf[0])             # confidence score
      
      if tb_bbox is None or conf > tb_bbox["conf"]:
        tb_bbox = {
          "bbox": (x1, y1, x2, y2),
          "conf": conf,
          "class_id": target_class
        }
  
  # Draw the bounding box for the toothbrush if detected
  if draw and tb_bbox is not None:
    x1, y1, x2, y2 = map(int, tb_bbox["bbox"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Toothbrush: {tb_bbox['conf']:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
  
  return tb_bbox

# utils for hands
def average_lms(lms):
  avg_x = sum(lm.x for lm in lms) / len(lms)
  avg_y = sum(lm.y for lm in lms) / len(lms)
  return (avg_x, avg_y)

def draw_lm(img, lm):
  h, w, _ = img.shape
  cx, cy = int(lm[0] * w), int(lm[1] * h)
  cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
  
def draw_line(img, lm1, lm2):
  h, w, _ = img.shape
  cx1, cy1 = int(lm1[0] * w), int(lm1[1] * h)
  cx2, cy2 = int(lm2[0] * w), int(lm2[1] * h)
  cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)

def hand_predict(frame, draw=False, hands=hands):
  # hands inference
  results = hands.process(frame)

  if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
      wrist = (handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[0].z)
      mid = average_lms([handLms.landmark[5], handLms.landmark[9], handLms.landmark[13], handLms.landmark[17]])
      tip = average_lms([handLms.landmark[8], handLms.landmark[12], handLms.landmark[16], handLms.landmark[20], handLms.landmark[4]])
      
      if draw:
        draw_lm(frame, wrist)
        draw_lm(frame, mid)
        draw_lm(frame, tip)
        draw_line(frame, wrist, mid)
        draw_line(frame, mid, tip)
        
      return (wrist, mid, tip)
  return None


def mouth_predict(frame, draw=False, moondream=moondream):
  image = Image.fromarray(frame)
  encoded_image = moondream.encode_image(image)
  
  result = moondream.query(encoded_image, "Is the person in the image smiling with their teeth or do they have their mouth wide open? respond only with Smile or Open.")
  answer = result["answer"]
  is_smiling = answer == "Smile"

  settings = {"max_objects": 1}
  result = moondream.detect(encoded_image, "mouth", settings)
  detections = result["objects"]
  
  if draw and len(detections) > 0:
    x_min = detections[0]["x_min"] * image.width
    y_min = detections[0]["y_min"] * image.height
    x_max = detections[0]["x_max"] * image.width
    y_max = detections[0]["y_max"] * image.height
    
    # Draw bounding box on the frame
    color = (0, 255, 0) if is_smiling else (0, 0, 255)
    cv2.rectangle(
        frame,
        (int(x_min), int(y_min)),
        (int(x_max), int(y_max)),
        color,
        2
    )
  
    return (is_smiling, detections[0])
  return None


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    yolo = yolo_predict(frame, True)
    hands = hand_predict(frame, True)
    mouth = mouth_predict(frame, True)
    
    if yolo is not None and hands is not None and mouth is not None:
      # collect all the data
      tb_x1, tb_y1, tb_x2, tb_y2 = yolo["bbox"]
      tb_mid_x = (tb_x1 + tb_x2) / 2
      tb_mid_y = (tb_y1 + tb_y2) / 2
      
      hand_wrist, hand_mid, hand_tip = hands
      hw_x, hw_y = hand_wrist[0] * frame_width, hand_wrist[1] * frame_height
      hm_x, hm_y = hand_mid[0] * frame_width, hand_mid[1] * frame_height
      ht_x, ht_y = int(hand_tip[0] * frame_width), hand_tip[1] * frame_height
      
      is_smiling, mouth_bbox = mouth
      m_x1 = mouth_bbox["x_min"] * frame_width
      m_y1 = mouth_bbox["y_min"] * frame_height
      m_x2 = mouth_bbox["x_max"] * frame_width
      m_y2 = mouth_bbox["y_max"] * frame_height
      mouth_mid_x = (m_x1 + m_x2) / 2
      mouth_mid_y = (m_y1 + m_y2) / 2
      
      # center the data to the origin (relative to the mouth)
      delta_x = -mouth_mid_x
      delta_y = -mouth_mid_y

      tb_mid_x += delta_x
      tb_mid_y += delta_y
      
      hw_x += delta_x
      hw_y += delta_y
      hm_x += delta_x
      hm_y += delta_y
      ht_x += delta_x
      ht_y += delta_y
      
      mouth_mid_x += delta_x
      mouth_mid_y += delta_y
      
      row = {
        "tb_mid_x": tb_mid_x, "tb_mid_y": tb_mid_y,
        "hw_x": hw_x, "hw_y": hw_y,
        "hm_x": hm_x, "hm_y": hm_y,
        "ht_x": ht_x, "ht_y": ht_y,
        "is_smiling": is_smiling
      }
      
      res = predict_one(
        row=row,
        model=model,
        scaler_mean=mean,
        scaler_scale=scale,
        feature_cols=DEFAULT_FEATURE_COLS,
        use_relative_diffs=False,
        device="cpu",
      )
      print(res)
      
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()