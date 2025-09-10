# Object Segmentation & Human Blur

## Task Description

This project implements **automatic human detection and anonymization** in images using **YOLOv8 segmentation** and face detection. The app detects humans (faces/bodies) and applies a blur to protect identity, while leaving the background untouched. Key features:

- **YOLOv8 segmentation (`yolov8n-seg.pt`) for full-body human detection**
- **YOLOv8 face detection (`yolov8n-face.pt`) for face-only anonymization**
- **Dual mode**: blur both faces and bodies for stronger anonymization
- **Gaussian or pixelation blur**, applied only to detected regions
- **Adjustable blur strength** via slider
- **Parallel processing** of multiple images using configurable tasks
- **Metadata/logs** showing number of humans detected, faces detected, and processing time
- **Download button** for each processed image
- **Supports multiple image formats**: jpg, jpeg, png, bmp, tiff

The goal is to provide a **simple, interactive and efficient tool for privacy protection** in images with flexible detection modes.

---

## Project Structure

HackP2025/
│── task6_object_segmentation_blur/
│   ├── README.md
│   ├── requirements.txt
│   ├── segment_and_blur.py
│   ├── yolov8n-seg.pt
│   ├── yolov8n-face.pt
│   └── samples/
│       └── sample_queries/

---

## Installation

It is recommended to use a **virtual environment**.

### Create and activate venv

**Linux / Mac**

```
python3 -m venv venv
source venv/bin/activate

```

**On Windows**

```
python -m venv venv
.\venv\Scripts\activate

```

## Install dependencies

pip install -r requirements.txt

### YOLO Models

This project uses **two YOLOv8 models**:

1. **Full-body segmentation (`yolov8n-seg.pt`)**  
   - Required for **Full Body** or **Dual** detection modes.  
   - Automatically downloaded during initial run (included in the repository).  
   - [Download link from Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt)

2. **Face detection (`yolov8n-face.pt`)**  
   - Required for **Face Only** or **Hybrid** detection modes.  
   - Must be downloaded manually (included in the repository).  
   - [Download link](https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt)

> **Note:** The models are **already included in this repository** (`yolov8n-seg.pt` and `yolov8n-face.pt`).  

> If you want more advanced versions, you can download them from the links above and place them in the same folder as `segment_and_blur.py`.

---

## Usage

Run the Streamlit app to upload images and automatically blur humans:

```
streamlit run segment_and_blur.py

```

- Upload one or multiple images (jpg, jpeg, png, bmp, tiff) 
 
- Select **detection mode**: Full body, Face only, or Both 
 
- Select **blur style** (Gaussian or Pixelation)  
 
- Adjust **parallel tasks** for faster batch processing

- Adjust **blur strength** with the slider  

- Preview original and blurred images side by side 
 
- View **metadata/logs** showing humans detected, faces detected, and processing time 
 
- Download each blurred image individually

---

### Observations

- **Face-only blur**: Precise anonymization for sensitive images.  
- **Full-body blur**: Standard human anonymization.  
- **Dual mode**: Extra protection for both faces and bodies.  
- **Adjustable blur strength**: Higher values ensure strong anonymization for high-resolution images.  
- **Pixelation optimized**: Works reliably across different image sizes.  
- **Parallel processing: Faster processing for multiple images
- **Metadata/logs**: Provides real-time feedback on detection counts and processing time.  
- **Multiple uploads**: Users can process several images in one session.

---

## Possible Extensions

Batch processing and exporting multiple images at once.  

Integration with **video input** for automatic human blurring in videos.
  
Optional **mask overlay preview** before applying blur. 
 
Web/desktop deployment beyond Streamlit.
  
---

## Author

**Akhil V Nair** – HackP 2025