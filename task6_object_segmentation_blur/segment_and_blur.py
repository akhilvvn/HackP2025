import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Object Segmentation & Human Blur", layout="wide")
st.title("Object Segmentation & Human Blur")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "bmp", "tiff"], accept_multiple_files=True)
detection_mode = st.selectbox("Detection mode", ["Full body", "Face only", "Both"])
blur_style = st.selectbox("Blur style", ["Gaussian", "Pixelation"])
blur_strength = st.slider("Blur strength", 5, 101, 25)
max_workers = st.slider("Parallel tasks", 1, 8, 4)

@st.cache_resource
def load_models(mode):
    body = YOLO("yolov8n-seg.pt") if mode in ["Full body", "Both"] else None
    face = YOLO("yolov8n-face.pt") if mode in ["Face only", "Both"] else None
    return body, face

if uploaded_files:
    with st.spinner("Loading models..."):
        body_model, face_model = load_models(detection_mode)

    def create_face_mask(shape, boxes):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(mask, (int(cx), int(cy)), (int(axes[0]), int(axes[1])), 0, 0, 360, 1, -1)
        return mask.astype(bool)

    def gaussian(img, mask, ksize):
        k = ksize if ksize % 2 == 1 else ksize + 1
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        out = img.copy()
        out[mask_3c] = blurred[mask_3c]
        return out

    def pixelate(img, mask, blocks=10):
        img_out = img.copy()
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return img_out
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        roi = img[y1:y2 + 1, x1:x2 + 1].copy()
        h, w = roi.shape[:2]
        down_w, down_h = max(1, w // blocks), max(1, h // blocks)
        small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_mask = mask[y1:y2 + 1, x1:x2 + 1]
        roi_mask_3c = np.repeat(roi_mask[:, :, np.newaxis], 3, axis=2)
        roi[roi_mask_3c] = pixelated[roi_mask_3c]
        img_out[y1:y2 + 1, x1:x2 + 1] = roi
        return img_out

    def process_file(uploaded_file):
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(image)
        img_out = img_array.copy()
        human_count, face_count = 0, 0
        start_time = time.time()
        combined_mask = np.zeros(img_array.shape[:2], dtype=bool)

        if body_model:
            results = body_model(img_array)
            if len(results):
                r = results[0]
                if hasattr(r, "masks") and r.masks is not None:
                    masks = r.masks.data.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    for m, cls in zip(masks, classes):
                        if int(cls) == 0:
                            human_count += 1
                            mask_resized = cv2.resize((m >= 0.5).astype(np.uint8), (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                            combined_mask |= mask_resized

        if face_model:
            results_f = face_model(img_array)
            if len(results_f):
                r = results_f[0]
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes.xyxy):
                    boxes = r.boxes.xyxy.cpu().numpy().reshape(-1, 4).tolist()
                    face_count += len(boxes)
                    face_mask = create_face_mask(img_array.shape, boxes)
                    combined_mask |= face_mask

        if blur_style == "Gaussian":
            img_out = gaussian(img_out, combined_mask, blur_strength)
        else:
            blocks = max(1, blur_strength // 2)
            img_out = pixelate(img_out, combined_mask, blocks)

        processing_time = round(time.time() - start_time, 2)
        buf = io.BytesIO()
        Image.fromarray(img_out).save(buf, format="PNG")
        buf.seek(0)
        return {
            "name": uploaded_file.name,
            "processed_pil": Image.fromarray(img_out),
            "buf": buf,
            "human_count": human_count,
            "face_count": face_count,
            "processing_time": processing_time
        }

    files = list(uploaded_files)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f.name for f in files}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"name": futures[fut], "error": str(e)})

    for res in results:
        if "error" in res:
            st.error(f"Error processing {res['name']}: {res['error']}")
            continue
        st.image(res["processed_pil"], caption=f"Blurred: {res['name']}", use_container_width=True)
        st.markdown(f"**File:** {res['name']} | **Humans detected:** {res['human_count']} | **Faces detected:** {res['face_count']} | **Processing time:** {res['processing_time']}s")
        download_name = f"{Path(res['name']).stem}_blurred.png"
        st.download_button("Download Blurred Image", data=res["buf"], file_name=download_name, mime="image/png")

