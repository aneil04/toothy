import cv2
from transformers import AutoModelForCausalLM
from PIL import Image
import torch

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda", # "cuda" on Nvidia GPUs
)

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        break
    image = Image.fromarray(frame)
    encoded_image = model.encode_image(image)
    
    result = model.query(encoded_image, "Is the person in the image smiling with their teeth or do they have their mouth wide open? respond only with Smile or Open.")
    answer = result["answer"]
    print(f"Answer: {answer}")
    
    result = model.detect(encoded_image, "toothbrush")
    detections = result["objects"]
    
    for obj in detections:
      x_min = obj["x_min"] * image.width
      y_min = obj["y_min"] * image.height
      x_max = obj["x_max"] * image.width
      y_max = obj["y_max"] * image.height
      
      # Draw bounding box on the frame
      cv2.rectangle(
          frame,
          (int(x_min), int(y_min)),
          (int(x_max), int(y_max)),
          (0, 255, 0),
          2
      )
      cv2.putText(
          frame,
          obj.get("label", "toothbrush"),
          (int(x_min), int(y_min) - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 255, 0),
          2,
          cv2.LINE_AA
      )
      
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()