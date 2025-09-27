import cv2
import numpy as np
import mediapipe as mp

mp_fd = mp.solutions.face_detection
mp_fm = mp.solutions.face_mesh

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
            lm = res.multi_face_landmarks[0].landmark
            # Collect lip landmark indices from the predefined FACEMESH_LIPS connections
            lip_idx = np.unique(np.array(list(mp_fm.FACEMESH_LIPS)).flatten())
            lip_pts = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in lip_idx])

            # Mouth bbox
            x1, y1 = lip_pts.min(axis=0)
            x2, y2 = lip_pts.max(axis=0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # Very simple “mouth open” ratio (height/width of lip bbox)
            mouth_open_ratio = (y2 - y1) / max(1, (x2 - x1))
            mouth_open = mouth_open_ratio > 0.5
            if mouth_open:             
              cv2.putText(frame, f"Open: {mouth_open_ratio:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
              cv2.putText(frame, f"Close: {mouth_open_ratio:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
              
        cv2.imshow("cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
