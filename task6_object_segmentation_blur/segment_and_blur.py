import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import time

st.set_page_config(page_title="Object Segmentation & Human Blur")
st.title("Object Segmentation & Human Blur")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "bmp", "tiff"], accept_multiple_files=True)
detection_mode = st.selectbox("Detection mode", ["Full body", "Face only", "Both"])
blur_style = st.selectbox("Blur style", ["Gaussian", "Pixelation"])
blur_strength = st.slider("Blur strength", 5, 101, 25)

def pixelate_region(img, mask, blocks=10):
    masked_img = img.copy()
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        return img
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    roi = masked_img[y1:y2+1, x1:x2+1]
    h, w = roi.shape[:2]
    block_h = max(1, h // blocks)
    block_w = max(1, w // blocks)
    temp = cv2.resize(roi, (block_w, block_h), interpolation=cv2.INTER_LINEAR)
    roi_pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    roi_pixelated_masked = roi_pixelated.copy()
    roi_pixelated_masked[~mask[y1:y2+1, x1:x2+1]] = roi[~mask[y1:y2+1, x1:x2+1]]
    masked_img[y1:y2+1, x1:x2+1] = roi_pixelated_masked
    return masked_img

def detect_faces(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    mask = np.zeros(img_array.shape[:2], dtype=bool)
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = True
    return mask, len(faces)

if uploaded_files:
    model = YOLO("yolov8n-seg.pt") if detection_mode != "Face only" else None
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        start_time = time.time()
        human_count = 0
        face_count = 0
        combined_mask = np.zeros(img_array.shape[:2], dtype=bool)

        if detection_mode in ["Full body", "Both"]:
            results = model(img_array)[0]
            masks = results.masks
            if masks is not None:
                for mask, cls in zip(masks.data, results.boxes.cls):
                    if int(cls) == 0:
                        human_count += 1
                        mask_resized = cv2.resize(mask.cpu().numpy().astype(np.uint8),
                                                  (img_array.shape[1], img_array.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST)
                        combined_mask |= mask_resized.astype(bool)

        if detection_mode in ["Face only", "Both"]:
            face_mask, faces_detected = detect_faces(img_array)
            face_count = faces_detected
            combined_mask |= face_mask

        if np.any(combined_mask):
            if blur_style == "Gaussian":
                kernel_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
                blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
                img_array[combined_mask] = blurred[combined_mask]
            else:
                img_array = pixelate_region(img_array, combined_mask, blocks=max(1, blur_strength // 2))

        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        st.write(f"**File:** {uploaded_file.name} | **Humans detected:** {human_count} | **Faces detected:** {face_count} | **Processing time:** {processing_time}s")
        st.image([image, Image.fromarray(img_array)], width=300, caption=["Original", "Blurred"])

        buf = io.BytesIO()
        Image.fromarray(img_array).save(buf, format="PNG")
        st.download_button("Download Blurred Image", buf, file_name=f"blurred_{uploaded_file.name}", mime="image/png")
