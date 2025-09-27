import cv2
import numpy as np
import mediapipe as mp

mp_fd = mp.solutions.face_detection
mp_fm = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_face_landmarks(frame, face_landmarks, width, height):
    """Draw all face landmarks and connections."""
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_fm.FACEMESH_TESSELATION,   # full mesh
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
    )

cap = cv2.VideoCapture(0)
with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd, \
     mp_fm.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:

    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Face box (optional – useful to skip mesh when no face)
        fdet = fd.process(rgb)
        if not fdet.detections:
            cv2.imshow("cam", frame)
            if cv2.waitKey(1) == 27: break
            continue

        # Landmarks (gives lips)
        res = fm.process(rgb)
        if res.multi_face_landmarks:
            h, w = frame.shape[:2]
            face_landmarks = res.multi_face_landmarks[0]
            draw_face_landmarks(frame, face_landmarks, w, h)
            
            # --- Calculate mouth bbox ---
            lip_idx = np.unique(np.array(list(mp_fm.FACEMESH_LIPS)).flatten())
            lip_pts = np.array([[int(face_landmarks.landmark[i].x*w),
                                 int(face_landmarks.landmark[i].y*h)] for i in lip_idx])
            x1, y1 = lip_pts.min(axis=0)
            x2, y2 = lip_pts.max(axis=0)
            mouth_height = y2 - y1
            
             # --- Calculate face bbox height (from detection box) ---
            det = fdet.detections[0]
            bbox = det.location_data.relative_bounding_box
            face_ymin = int(bbox.ymin * h)
            face_ymax = int((bbox.ymin + bbox.height) * h)
            face_height = face_ymax - face_ymin

            # --- Normalized mouth-open ratio ---
            mouth_open_ratio = mouth_height / max(1, face_height)

            # Optional: threshold to classify open/closed
            if mouth_open_ratio > 0.235:  # <-- tune threshold per dataset/camera
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
                
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
              
        cv2.imshow("cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
