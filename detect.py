import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

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

# setup opencv
cap = cv2.VideoCapture(0)

# draw all face landmarks and connections.
def draw_face_landmarks(frame, face_landmarks, mp_fm=mp_fm, mp_drawing=mp_drawing, mp_styles=mp_styles):
  mp_drawing.draw_landmarks(
    image=frame,
    landmark_list=face_landmarks,
    connections=mp_fm.FACEMESH_TESSELATION,   # full mesh
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
  )

# get bounding boxes for toothbrush
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

def hand_predict(frame, draw=False, hands=hands):
  # hands inference
  results = hands.process(frame)

  if results.multi_hand_landmarks:
    if draw:
      for handLms in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    return results.multi_hand_landmarks

  return None

def mouth_predict(frame, draw=False, mp_fd=mp_fd, mp_fm=mp_fm):
  with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd, mp_fm.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
    # Face box (optional – useful to skip mesh when no face)
    fdet = fd.process(frame)
    if not fdet.detections:
      return None   
    
    # Landmarks (gives lips)
    res = fm.process(frame)
    if res.multi_face_landmarks:
      h, w = frame.shape[:2]
      face_landmarks = res.multi_face_landmarks[0]
      
      # --- Calculate mouth bbox ---
      lip_idx = np.unique(np.array(list(mp_fm.FACEMESH_LIPS)).flatten())
      lip_pts = np.array([[int(face_landmarks.landmark[i].x*w), int(face_landmarks.landmark[i].y*h)] for i in lip_idx])
      x1, y1 = lip_pts.min(axis=0)
      x2, y2 = lip_pts.max(axis=0)
      mouth_height = y2 - y1
      mouth_center_x = (x1 + x2) / 2
      mouth_center_y = (y1 + y2) / 2
      
        # --- Calculate face bbox height (from detection box) ---
      det = fdet.detections[0]
      bbox = det.location_data.relative_bounding_box
      face_ymin = int(bbox.ymin * h)
      face_ymax = int((bbox.ymin + bbox.height) * h)
      face_height = face_ymax - face_ymin

      # --- Normalized mouth-open ratio ---
      mouth_open_ratio = mouth_height / max(1, face_height)
      mouth_open = mouth_open_ratio > 0.235
      
      if draw:    
        if mouth_open:
          color = (0, 255, 0)
        else:
          color = (0, 0, 255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        draw_face_landmarks(frame, face_landmarks)
      
      return (mouth_open, mouth_center_x, mouth_center_y)
    return None
      

while cap.isOpened():
  # capture frame
  success, frame = cap.read()
  if not success:
    break
  frame = cv2.flip(frame, 1)
  
  # run inference
  yolo = yolo_predict(frame, True)
  hands = hand_predict(frame, True)
  mouth = mouth_predict(frame, True)  

  # show frame
  cv2.imshow("Toothbrush Detection", frame)

  # verify we can detect the toothbrush, mouth, and hands
  if yolo is not None and mouth is not None and hands is not None:
    # unpack the results
    tb_bbox = yolo
    mouth_open, mouth_center_x, mouth_center_y = mouth
    
    print(tb_bbox, mouth_open, mouth_center_x, mouth_center_y)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
cap.release()
cv2.destroyAllWindows()