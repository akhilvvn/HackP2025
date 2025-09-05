# Search Images with Image

## Task Description

This project implements an **image-to-image search** using **OpenAI’s CLIP model**. Given a query image uploaded by the user, it retrieves the most visually and semantically similar images from a local dataset of animal images. The system implements:

- **CLIP (ViT-B/32) for zero-shot image-image similarity**
- **Cosine similarity for ranking results**
- **Similarity thresholding to filter low-confidence matches**
- **Side-by-side preview generation of top matches**
- **Saves top matches along with similarity scores in `results.csv`**
- **Supports GPU acceleration if available**

The goal is to allow interactive retrieval of the most relevant images from a small, pre-defined dataset based on visual similarity.

---

## Project Structure

HackP2025/
│── task5_search_images_with_image/
│   ├── README.md
│   ├── requirements.txt
│   ├── image_search.py
│   ├── samples/
│   │   ├── dataset/          # ~50 animal images
│   │   └── sample_queries/   # optional test input images
│   └── outputs/
│       ├── results.csv     # top matches for each query
│       └── previews/       # side-by-side preview images
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

Run the Streamlit app to upload a query image and get top matches:
```

streamlit run image_search.py

```

Upload a query image (jpg, jpeg, png, webp)

Top matching images from the dataset will be displayed side-by-side

Results are saved to outputs/results.csv

Preview images are saved in outputs/previews/.

---


### Outputs will be saved in the outputs/ folder:

results.csv → Ranked top matches for each query along with similarity scores

previews/ → Top matching images copied into per-query subfolders

---

## Sample Results

From `results.csv`:

| Query Image    | Rank | Matched Filename | Similarity |
| -------------- | ---- | ---------------- | ---------- |
| 9b3f016199.jpg | 1    | 0fe508ab40.jpg   | 0.8940     |
| 9b3f016199.jpg | 2    | 0a47b7d021.jpg   | 0.8567     |
| 2eb0feaa3a.jpg | 1    | 2dab1e3035.jpg   | 0.7844     |
| 2eb0feaa3a.jpg | 2    | 1a716cd8b3.jpg   | 0.7701     |
| 9efd18dd6c.jpg | 1    | 7c43d5ca9e.jpg   | 0.8750     |
| 9efd18dd6c.jpg | 2    | 4aacd195b5.jpg   | 0.7952     |

---

### Observations

- **Small dataset:** Limited images per species limits retrieval diversity.  
- **Similarity threshold:** Set at optimal to reduce irrelevant or low-confidence matches.  
- **CLIP embeddings:** Even with a small dataset, embeddings provide reasonable semantic similarity.  
- **Top matches:** Retrieved images above the threshold generally correspond to the correct species.  
- **Side-by-side previews:** Help visually verify the similarity between the query and dataset images.  
- **Limitations:** Visually similar animals (e.g., lion, tiger, leopard) can sometimes appear together due to embedding proximity.

While current results are satisfactory, a **larger and more diverse dataset** is needed to ensure robust performance and better generalization.

---

## Possible Extensions

Increase dataset size for further robustness.

Fine-tuning CLIP or few-shot/adapter training.

Clustering for post-processing redundant matches if the dataset is large enough.

Web/desktop deployment beyond the current Streamlit interface.

Enhanced previews with multiple top matches or annotated similarity scores (currently just side-by-side images).

---

## Author

**Akhil V Nair** – HackP 2025