# Hac'KP 2025 – Task Submissions

---

## Author

**Name:** Akhil V Nair

**Email:** akhilvvnair@gmail.com

**College:** Lourdes Matha College of Science and Technology, Thiruvananthapuram

---

## Overview

This repository contains **8 projects** completed as part of **Hac'KP 2025**.

Each project explores AI/ML applications in **image search, object detection, privacy protection, and investigative tools**.

## List of Tasks Attempted

1. **Image Similarity Scoring** — perceptual/difference hashes, SSIM, ORB matching, and a Streamlit UI for interactive testing and CSV export.

2. **Indoor/Outdoor Image Classifier** — ResNet18 fine-tune for 5 classes with weighted loss, augmentation, early stopping, and Streamlit UI.

3. **Image Metadata Analysis** — EXIF/GPS/Timestamp extraction, OCR (Tesseract), language detection, AI-generation heuristics, and visualizations.

4. **Search Images with Text** — CLIP (ViT-B/32) text→image semantic search with caching, clustering, and Streamlit UI.

5. **Search Images with Image** — CLIP image→image retrieval (upload query image → ranked results + previews).

6. **Object Segmentation & Human Blur** — YOLOv8 segmentation + face detection for anonymization (blur/pixelate), with Streamlit UI.

7. **Search Objects With Image Query** — object-level retrieval using YOLO + CLIP embeddings (detect objects → retrieve dataset images with same object).

8. **Investigator Web Tool** — FastAPI backend + React (Vite) + Konva frontend for manual & AI-assisted image redaction (blackout/blur/crop).

---

## Quick Notes

- It is **recommended to use virtual environments** (venv) for Python-based projects.

- Most GUIs use **Streamlit** (local web UI).

- Some tasks include large model files (YOLO weights); see **Large Files & Models** below.

- Each subfolder contains a detailed README and a requirements.txt (or package.json for the frontend).

- If a task requires additional system setup (e.g., Tesseract), it's noted under that task.

---

## How to run each task (paths, commands, dependencies)

**Replace source venv/bin/activate with .\venv\Scripts\activate on Windows**

#### 1. Image Similarity Scoring

**Path:** task1_image_similarity_scoring/

```

cd task1_image_similarity_scoring

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt


# CLI mode

python similarity_scoring.py

# Streamlit UI

streamlit run similarity_scoring.py streamlit

```

**Outputs:** outputs/scores.csv, outputs/scores_full.csv, outputs/orb_matches/

**Dependencies)**: Pillow, imagehash, scikit-image, opencv-python, matplotlib, numpy, pandas, streamlit, faiss-cpu, torch, torchvision, transformers

#### 2. Indoor/Outdoor Image Classifier

**Path**: task2_indoor_outdoor_classifier/

```

cd task2_indoor_outdoor_classifier

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt


# Train

python classifier.py --mode train

# CLI inference

python classifier.py --mode cli

# Streamlit UI

streamlit run classifier.py

```

**Outputs:** outputs/predictions.csv, outputs/plots/ (confusion matrices)

**Dependencies:** torch, torchvision, numpy, pillow, matplotlib, scikit-learn, streamlit

#### 3. Image Metadata Analysis

**Path**: task3_image_metadata_analysis/

```

cd task3_image_metadata_analysis

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt


# Streamlit UI

streamlit run metadata_extractor.py

```

**Notes / Additional setup (OCR):**

Tesseract OCR must be installed on the system and available in PATH for pytesseract to work.

**Windows:** Use the UB Mannheim build (installer) — add to PATH. Or set pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" in metadata_extractor.py (already included).

**Ubuntu/Debian:** sudo apt install tesseract-ocr

**macOS:** brew install tesseract

**Outputs:** CSV/JSON metadata exports, visualizations.

**Dependencies:** streamlit, pandas, Pillow, exifread, pytesseract, matplotlib, numpy, langid

#### 4. Search Images with Text

**Path**: task4_search_images_with_text/

```

cd task4_search_images_with_text

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt


# Streamlit UI

streamlit run text_search.py

```

**Behavior:** Computes/loads cached CLIP embeddings for dataset images (embeddings/), supports similarity threshold & clustering, and writes results.csv.

**First run:** May download pretrained CLIP model & compute embeddings (takes time).

**Dependencies:** torch, torchvision, transformers, pillow, pandas, matplotlib, scikit-learn, streamlit, numpy

#### 5. Search Images with Image

**Path**: task5_search_images_with_image/

```

cd task5_search_images_with_image

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt


# Streamlit UI

streamlit run image_search.py

```

**Outputs:** outputs/results.csv, outputs/previews/

**Dependencies:** torch, torchvision, transformers, pandas, scikit-learn, Pillow, streamlit, numpy

#### 6. Object Segmentation & Human Blur

**Path**: task6_object_segmentation_blur/

```

cd task6_object_segmentation_blur

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

# Streamlit UI

streamlit run segment_and_blur.py

```

**Models:** yolov8n-seg.pt and yolov8n-face.pt (included in repo). If missing, download from:

yolov8n-seg.pt — [Download link from Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt)

yolov8n-face.pt — [Lindevs release for face detection](https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt)

**Outputs:** Download blurred images (downloadable), logs/metadata.

**Dependencies:** ultralytics, opencv-python, Pillow, numpy, streamlit

#### 7. Search Objects With Image Query

**Path**: task7_search_objects_with_image/

```

cd task7_search_objects_with_image

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt


# Streamlit UI

streamlit run app.py

```

**Behavior:** Detects objects in query image (YOLO), computes CLIP embeddings for object crops, and retrieves dataset images containing same/similar objects. Results and object crops/embeddings cached under outputs/dataset_objects/.

**Notes on YOLO11x:** YOLO11x model weights are large (>100 MB). The project includes lighter yolo11l.pt for convenience. If you want yolo11x.pt download and replace in project root and update object_search.py as instructed in the project README.

**Dependencies:** torch, torchvision, torchaudio, pillow, ultralytics, sentence-transformers, tqdm, numpy

#### 8. Investigator Web Tool (FastAPI backend + React frontend)

**Path**: task8_investigator_web_tool/

**Backend (FastAPI)**

```

cd task8_investigator_web_tool/backend

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt


# Start server

uvicorn main:app --reload

```

Backend runs at: http://127.0.0.1:8000 by default.

**Backend requirements (provided):** fastapi, uvicorn[standard], pillow, opencv-python, numpy, torch, torchvision, timm, requests, python-multipart

**Frontend (React + Vite + Konva)**

```

cd task8_investigator_web_tool/frontend

npm install

npm run dev

```

Frontend runs at: http://localhost:5173 by default.

**Ensure Node.js v18+ (or newer) is installed**

**Frontend deps (from package.json):** axios, konva, react, react-dom, react-dropzone, react-icons, react-konva, use-image

Dev: @vitejs/plugin-react, vite

**Note:** YOLO weights (yolov8l.pt) are located in the backend folder for server-side AI-assisted detection. Ultralytics will auto-download on first run if missing.

---

## Large Files & Models

- **YOLO Weights**

  - Task 6 → yolov8n-seg.pt, yolov8n-face.pt (both included in repo)

  - Task 7 → yolo11l.pt (included), with optional manual upgrade to yolo11x.pt (download link in sub-README)

  - Task 8 → yolov8l.pt (downloaded automatically if missing)

- Pretrained CLIP / transformer models are downloaded automatically on first run (internet required).

- Tesseract OCR requires system-level installation. Please refer to Task 3 section for instructions.

- If any large files or dependencies are required but not included, they are explicitly noted in the respective task README along with download/setup instructions.

---

## Sample Data & Caching

- **Sample Datasets**

  - Each task folder under samples/ contains a small dataset for testing and demonstration.

  - Examples:

  - Task 1–5 → small curated image sets (~50 images each, e.g., animals, objects, or OCR text samples).

  - Task 7 → a larger dataset (~400 images) used for object retrieval and similarity search.

  - These datasets are intentionally lightweight to keep the repository size manageable while still showing the functionality of each project.

- **Caching of Computed Features**

  - Some projects (e.g., Task 2 – clustering, Task 7 – object retrieval) pre-compute embeddings or object crops during the **first run**.

  - Results are saved locally into folders such as embeddings/ or outputs/dataset_objects/.

  - On subsequent runs, the system **loads cached embeddings instead of recomputing them**, making it much faster to experiment or re-run queries.

- **Using Your Own Data**

  - You can replace or extend the sample datasets with your own images.

  - If you add a new dataset, simply place it in the respective samples/ folder. Caches will be rebuilt automatically during the next run.

---

## Demo Videos

### Individual Task Demos

- **Task 1 – Image Similarity Scoring** → [Watch here](https://youtu.be/iNFG1b_CkA8)

- **Task 2 – Indoor/Outdoor Image Classifier** → [Watch here](https://youtu.be/GBSF0oX78cM)

- **Task 3 – Image Metadata Analysis** → [Watch here](https://youtu.be/c9NtQ6CEmUw)

- **Task 4 – Search Images with Text** → [Watch here](https://youtu.be/rTTIsdoOjw4)

- **Task 5 – Search Images with Image** → [Watch here](https://youtu.be/FsfuN38bC5U)

- **Task 6 – Object Segmentation & Human Blur** → [Watch here](https://youtu.be/WaI0WZYmd7E)

- **Task 7 – Search Objects With Image Query** → [Watch here](https://youtu.be/doNhZeksUiQ)

- **Task 8 – Investigator Web Tool** → [Watch here](https://youtu.be/Nqk8l-Wpl9I)

---

## Summary

- This repository is a collection of **mini-projects** built as part of **Hac'KP 2025**, showcasing applications of computer vision, deep learning, and interactive tools.

- Each task is **self-contained** with its own README.md, requirements, and sample data to make it easy to run individually.

- The included datasets are small and meant only for demonstration. For serious use cases, larger and more diverse datasets are recommended.

- Some tasks download pretrained models automatically on first run (CLIP, transformers, etc.). Others (Tesseract OCR, Node.js frontend) may require manual setup — see the individual task documentation.

- All projects are built with **reproducibility in mind** — cached embeddings, clustering results, and object crops are stored locally to make subsequent runs faster.

- The overall goal is not just functionality, but to show how **AI + interactivity** can be combined into investigative, privacy, and search workflows.

---

This repo is designed for learning, experimenting, and demonstrating — feel free to explore, adapt, and extend each project.