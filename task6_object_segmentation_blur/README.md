# Object Segmentation & Human Blur

## Task Description

This project implements **automatic human detection and anonymization** in images using **YOLOv8 segmentation** and face detection. Given any image uploaded by the user, it detects humans (faces/bodies) and applies a blur to protect identity, while leaving the background untouched. The system implements:

- **YOLOv8 segmentation (`yolov8n-seg`) for full-body human detection**
- **OpenCV face detection for face-only anonymization**
- **Hybrid mode**: blur both faces and bodies with stronger anonymization
- **Gaussian or pixelation blur** applied only to detected regions
- **Adjustable blur strength**
- **Side-by-side preview of original and blurred images**
- **Metadata/logs showing number of humans detected, faces detected, and processing time**
- **Download button for processed images**
- **Supports multiple image formats** (jpg, jpeg, png, bmp, tiff)

The goal is to provide a **simple and interactive tool for privacy protection** in images with flexible detection modes.

---

## Project Structure

HackP2025/
│── task6_object_segmentation_blur/
│   ├── README.md
│   ├── requirements.txt
│   ├── segment_and_blur.py
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

# Install dependencies

pip install -r requirements.txt

---

## Usage

Run the Streamlit app to upload images and automatically blur humans:
```

streamlit run segment_and_blur.py

```

- Upload one or multiple images (jpg, jpeg, png, bmp, tiff) 
 
- Select **detection mode**: Full body, Face only, or Both 
 
- Select **blur style** (Gaussian or Pixelation)  

- Adjust **blur strength** with the slider  

- Preview original and blurred images side by side 
 
- View **metadata/logs** showing humans detected, faces detected, and processing time 
 
- Download each blurred image individually

---

## Sample Results

| Uploaded Image        | Humans Detected | Faces Detected | Processing Time (s) |
| -------------------- | --------------- | -------------- | ----------------- |
| sample1.jpg           | 3               | 2              | 1.45              |
| sample2.png           | 5               | 5              | 1.98              |
| sample3.jpeg          | 1               | 1              | 0.87              |

Blurred images are available via the **download button** in the app.

---

### Observations

- **Face-only blur**: Precise anonymization for sensitive images.  
- **Full-body blur**: Standard human anonymization.  
- **Hybrid mode**: Extra protection for both faces and bodies.  
- **Adjustable blur strength**: Higher values ensure strong anonymization for high-resolution images.  
- **Pixelation optimized**: Works reliably across different image sizes.  
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