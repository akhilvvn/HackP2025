import streamlit as st
import cv2
import numpy as np
from utils import image_processing as ip
from utils import object_detection as od
from utils import helpers as hl
import os

st.set_page_config(page_title="Investigator Web Tool", layout="wide")

hl.ensure_dir("outputs/previews")
hl.ensure_dir("outputs")

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_image = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    tools_used = []

    ai_mode = st.sidebar.checkbox("AI-Assisted Detection")
    if ai_mode:
        od.load_model()
        bboxes, ai_mask, class_labels, confidences = od.detect_objects(image)
        st.session_state['ai_mask'] = ai_mask
        st.image(cv2.cvtColor(ai_mask, cv2.COLOR_GRAY2RGB), caption="AI Detected Objects")

    tool = st.sidebar.radio("Select Tool", ["Select", "Deselect", "Crop", "Blackout", "Blur"])
    intensity = st.sidebar.slider("Intensity", 1, 50, 15)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    coords_input = st.text_input("Enter Coordinates (x1,y1,x2,y2)")

    if st.button("Apply Tool") and coords_input:
        coords = tuple(map(int, coords_input.split(",")))
        if tool == "Select":
            mask = ip.select_region(mask, coords)
            tools_used.append("Select")
        elif tool == "Deselect":
            mask = ip.deselect_region(mask, coords)
            tools_used.append("Deselect")
        elif tool == "Crop":
            image = ip.crop_image(image, coords)
            original_image = image.copy()
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            tools_used.append("Crop")
        elif tool == "Blackout":
            image = ip.apply_blackout(image, mask, opacity=intensity/50)
            tools_used.append("Blackout")
        elif tool == "Blur":
            image = ip.apply_blur(image, mask, intensity=intensity)
            tools_used.append("Blur")

    if st.button("Preview"):
        preview = hl.generate_preview(original_image, image)
        st.image(preview, caption="Original vs Edited Preview", use_column_width=True)

    if st.button("Save Image"):
        save_path = os.path.join("outputs/previews", uploaded_file.name)
        hl.save_image(image, save_path)
        objects_edited = ["AI" if ai_mode else "Manual"]
        hl.log_edit("outputs/results.csv", uploaded_file.name, objects_edited, tools_used)
        st.success(f"Image saved to {save_path}")
