# Search Objects With Image Query

## Task Description

This project implements **object-level image retrieval** using **YOLOv8 object detection** and **CLIP embeddings**. Given any image uploaded by the user, the system detects objects and retrieves dataset images that contain the same object, even if it appears partially or off-center. The system implements:

- **YOLO11x for object detection**
- **CLIP (`clip-ViT-B-32`) embeddings for object similarity**
- **Class filtering**: search for specific object classes detected in the query image
- **Weighted similarity scoring** combining YOLO confidence and CLIP similarity
- **Adjustable similarity threshold**
- **Returns all matching images above threshold**
- **Grid layout of results with expandable full-size images**
- **Downloadable CSV report of matched images and similarity scores**
- **Supports multiple image formats** (jpg, jpeg, png)

The goal is to provide a **flexible and interactive tool for object-based image search**.

---

## Project Structure

HackP2025/
│── task7_search_objects_with_image/
│   ├── README.md
│   ├── requirements.txt
│   ├── app.py
│   ├── object_search.py
│   └── samples/
│       └── dataset/            #~400 images
│       └── sample_queries/     # optional query images
│   └── outputs/
│       └── dataset_objects/    # cached object crops and embeddings

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

## Key External Libraries and Models Used

- **ultralytics (YOLO11x)** for object detection. The **YOLO11x model** weights are too large to host directly on GitHub (>100 MB). On the first run, the script will attempt to download the model automatically but if the download fails,
  you can download the model directly from the link below and place the **yolo11x.pt** in the project root folder.

  **Download Link:** [YOLO11x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)

- **sentence-transformers (CLIP-ViT-B-32)** for embedding generation 
- **torch / torchvision** for tensor operations and GPU support  
- **Pillow** for image processing 
- **tqdm** for progress bars 
- **pandas** for CSV generation
- **streamlit** for GUI

**Note:** The **initial run may take several minutes** because:

1. YOLO11x object detection runs on every dataset image.
  
2. CLIP embeddings are computed for all detected objects.
 
3. Embeddings, object crops, confidence scores, and labels are saved in `outputs/dataset_objects/` for future runs, which will be much faster thanks to caching.

---

## Usage

Run the Streamlit app to upload query images and retrieve matching images from the dataset:
```

streamlit run app.py

```

- Upload one or multiple images (jpg, jpeg, png)
  
- The app automatically detects objects in the query image 
 
- Select an **object class** to search for  

- Adjust **similarity threshold** with the slider 
 
- View matching images in a **grid layout**  

- Click **Enlarge** to preview full-size images 
 
- Download a **CSV report** containing matched images and similarity scores

---

## Sample Results

| Query Image       | Selected Class | Match Rank | Matched Image        | Similarity Score |
| ----------------- | -------------- | ---------- | ------------------ | ---------------- |
| 000033 (7).jpg    | bed            | 1          | dataset\000030 (8).jpg | 0.680           |
| 000033 (7).jpg    | bed            | 2          | dataset\000036 (7).jpg | 0.669           |
| 000033 (7).jpg    | bed            | 3          | dataset\000041.png    | 0.638           |
| 000033 (7).jpg    | bed            | 4          | dataset\000039 (7).jpg | 0.615           |
| 000033 (7).jpg    | bed            | 5          | dataset\000034 (6).jpg | 0.610           |

All matched images are displayed in the app grid with similarity scores.

---

### Observations

- **Class filtering** reduces false positives  
- **Weighted similarity** ensures both detection confidence and visual similarity are considered  
- **Adjustable threshold** allows tuning between precision and recall  
- **Expandable images** provide detailed previews  
- **Supports partial or off-center objects** in retrieval  
- **CSV download** provides structured output for further analysis

---

## Possible Extensions

- Batch query processing for multiple images

- Integration with **larger datasets** and caching embeddings
  
- Multi-object queries: search for images containing multiple objects
  
- Web/desktop deployment for larger-scale usage
  
- Optional **visual bounding boxes** for matched objects in dataset images
    
---

## Author

**Akhil V Nair** – HackP 2025