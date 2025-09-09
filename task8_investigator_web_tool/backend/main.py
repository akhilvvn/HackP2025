from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageFilter, ImageDraw
import io
import numpy as np
from ultralytics import YOLO
import cv2
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('yolov8l.pt')

def apply_mask(image: Image.Image, mask: list, mode: str):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for r in mask:
        x, y, w, h = int(r['x']), int(r['y']), int(r['width']), int(r['height'])
        if mode == 'blackout':
            draw.rectangle([x, y, x + w, y + h], fill=(0,0,0))
        elif mode == 'blur':
            region = img.crop((x, y, x + w, y + h))
            region = region.filter(ImageFilter.GaussianBlur(radius=15))
            img.paste(region, (x, y))
    return img

@app.post("/blackout")
async def blackout(file: UploadFile = File(...), mask: str = Form(...)):
    image = Image.open(file.file).convert("RGB")
    mask_data = json.loads(mask)
    result = apply_mask(image, mask_data, mode='blackout')
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/blur")
async def blur(file: UploadFile = File(...), mask: str = Form(...)):
    image = Image.open(file.file).convert("RGB")
    mask_data = json.loads(mask)
    result = apply_mask(image, mask_data, mode='blur')
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/detect_objects")
async def detect_objects(file: UploadFile = File(...)):
    file_bytes = np.asarray(bytearray(file.file.read()), dtype=np.uint8)
    cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    results = model(cv_image)
    rectangles = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        rectangles.append({
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1)
        })
    return rectangles
