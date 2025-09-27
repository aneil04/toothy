import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.4)
mp_draw = mp.solutions.drawing_utils

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

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
          wrist = (handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[0].z)
          mid = average_lms([handLms.landmark[5], handLms.landmark[9], handLms.landmark[13], handLms.landmark[17]])
          tip = average_lms([handLms.landmark[8], handLms.landmark[12], handLms.landmark[16], handLms.landmark[20], handLms.landmark[4]])
          
          draw_lm(img, wrist)
          draw_lm(img, mid)
          draw_lm(img, tip)
          draw_line(img, wrist, mid)
          draw_line(img, mid, tip)

    cv2.imshow("Hands", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
