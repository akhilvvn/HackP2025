# Search Images with Text

## Task Description

This project implements a text-to-image search using **OpenAI's CLIP model**. Given a text query, it retrieves the most relevant images from a local **animal image dataset** based on semantic similarity. Additional features implemented:

- **CLIP (ViT-B/32) for zero-shot image-text similarity**
- **Cosine similarity for ranking results**
- **Similarity thresholding**
- **Clustering to filter out unrelated images**
- **Preview generation of top matches**
- **Saves top matches along with similarity scores in `results.csv`**
- **Supports GPU acceleration if available**

The goal is to enable natural language search on an image collection without explicit labeling or training, providing an interactive way to explore images semantically.

---

## Project Structure

HackP2025/
│── task4_search_images_with_text/
│   ├── README.md
│   ├── requirements.txt
│   ├── text_search.py
│   ├── samples/
│   │   └── dataset/        # ~50 animal images
│   └── outputs/
│       ├── results.csv     # top matches for each query
│       └── previews/       # result preview images

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

Run the search script on all images inside samples/dataset/:
```

python text_search.py

```

You will be prompted to enter text queries (eg. lion, crow, turtle etc.) Type `'exit'` to quit.

Note: The first run may take a few minutes depending on the dataset size and your hardware, because the script computes embeddings for all dataset images. On the first run, the pretrained CLIP model and processor will be downloaded from the internet. Subsequent runs will be faster as the model and embeddings are cached locally.

### Outputs will be saved in the outputs/ folder:

results.csv → Ranked top matches for each query along with similarity scores

previews/ → Top matching images copied into per-query subfolders

---

## Sample Results

From `results.csv`:

| Query   | Rank | Filename        | Similarity |
| ------- | ---- | --------------- | ---------- |
| lion    | 1    | 4dbdd0fea8.jpg  | 0.294      |
| lion    | 2    | 5fb837c61b.jpg  | 0.297      |
| dog     | 1    | 4aacd195b5.jpg  | 0.264      |
| dog     | 2    | 7c43d5ca9e.jpg  | 0.256      |
| cat     | 1    | 71756f7bd0.jpg  | 0.261      |
| cat     | 2    | 83e8a824a2.jpg  | 0.279      |
| tiger   | 1    | 2c75c39ece.jpg  | 0.266      |
| turtle  | 1    | 0a47b7d021.jpg  | 0.276      |
| turtle  | 2    | 0fe508ab40.jpg  | 0.280      |
| dolphin | 1    | 2dab1e3035.jpg  | 0.277      |
| crow    | 1    | 1d15117ae2.jpg  | 0.308      |


---

### Observations

- **Small dataset:** Limited images per class limits retrieval diversity.
- **Similarity threshold:** Set to optimal to reduce irrelevant matches.
- **CLIP embeddings:** Despite a small dataset, semantic matching works reasonably well.
- **Top matches:** All results above the threshold generally correspond to the correct species.
- **Limitations:** Similar animals (e.g., lion, tiger, leopard) can appear together due to embedding similarity.

While current results are satisfactory, a **larger and more diverse dataset** is needed to ensure robust performance and better generalization.

---

## Possible Extensions

Increase dataset size for further robustness.

Fine-tune or adapt CLIP on your dataset for higher accuracy.

Gradually unfreeze CLIP layers or train adapters for few-shot learning.

Post-processing: filtering and ranking to remove low-confidence or redundant matches.

Deploy the search interface as a web or desktop application.

---

## Author

**Akhil V Nair** – HackP 2025