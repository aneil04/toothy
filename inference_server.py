# server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from fastapi.responses import StreamingResponse
import time, uvicorn
from realtime_inference import predict_frame, detect_frame
import numpy as np
import cv2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    data = await image.read()
    pil = Image.open(BytesIO(data)).convert("RGB")
    
    # convert PIL to cv2
    frame = np.array(pil)
    frame = frame[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV
    frame = cv2.flip(frame, 1)
    
    preds = predict_frame(frame)

    return {} if preds is None else preds

@app.post("/detect")
async def infer(image: UploadFile = File(...)):
    data = await image.read()
    pil = Image.open(BytesIO(data)).convert("RGB")
    
    # convert PIL to cv2
    frame = np.array(pil)
    frame = frame[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV
    frame = cv2.flip(frame, 1)

    detect_frame(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
