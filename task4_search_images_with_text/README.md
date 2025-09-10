# Search Images with Text

## Task Description

This project implements a text-to-image search using **OpenAI's CLIP model**. Given a text query, it retrieves the most relevant images from a local **animal image dataset** based on semantic similarity. Additional features implemented:

- **CLIP (ViT-B/32) for zero-shot image-text similarity**
- **Cosine similarity for ranking results**
- **Similarity thresholding**
- **Clustering to filter out unrelated images**
- **Saves top matches along with similarity scores in `results.csv`**
- **Streamlit implementation for interactive search**
- **Supports GPU acceleration if available**
- **Caching of embeddings and clusters for faster subsequent runs**

The goal is to enable natural language search on an image collection without explicit labeling or training, providing an interactive way to explore images semantically.

---

## Project Structure

HackP2025/
│── task4_search_images_with_text/
│   ├── README.md
│   ├── requirements.txt
│   ├── text_search.py
│   ├── embeddings/             # cached embeddings and clusters
│   │   ├── image_embeddings.npy
│   │   └── clusters.npz
│   ├── samples/
│   │   ├── sample_queries.txt    # sample search queries
│   │   └── dataset/              # ~50 animal images

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

---

## Usage

Run the search script on all images inside `samples/dataset/` using Streamlit:

```

streamlit run text_search.py

```

This will open an interactive web interface in your browser where you can:

- Enter text queries (e.g., lion, crow, turtle, etc.)

- Adjust the similarity threshold slider

- Set the maximum number of results per query

- View top matching images directly in the interface

- Download the results as `results.csv`

Type `'exit'` in the query box to stop entering new queries.

Note: The first run may take a few minutes depending on the dataset size and your hardware, because the script computes embeddings for all dataset images and performs clustering. On the first run, the pretrained CLIP model and processor will also be downloaded from the internet. Subsequent runs will be faster as **embeddings and clusters are cached locally in `embeddings/`**.

---

## Sample Results

From `results.csv`:

| Query  | Rank | Filename        | Similarity |
| ------ | ---- | --------------- | ---------- |
| turtle | 1    | 0a47b7d021.jpg  | 0.276076   |
| turtle | 2    | 0fe508ab40.jpg  | 0.280161   |

---

### Observations

- **Small dataset:** Limited images per class limits retrieval diversity.
- **Similarity threshold:** Optimally set to reduce irrelevant matches.
- **CLIP embeddings:** Semantic matching works well despite the small dataset.
- **Top matches:** All results above the threshold generally correspond to the correct species.
- **Caching:** Embeddings and clusters are cached in `embeddings/` to speed up subsequent runs.
- **Streamlit UI:** Provides an interactive search interface with instant download of results.
- **Limitations:** Similar animals (e.g., lion, tiger, leopard) can appear together due to embedding similarity.

While current results are satisfactory, a **larger and more diverse dataset** is needed to ensure robust performance and better generalization.

---

## Possible Extensions

Increase dataset size for greater retrieval diversity and robustness.

Fine-tune or adapt CLIP on your dataset for higher accuracy.

Gradually unfreeze CLIP layers or train adapters for few-shot learning.

Implement additional post-processing: filtering and ranking to remove low-confidence or redundant matches.

Enhance the Streamlit UI with features like query history, similarity visualization, or dynamic clustering.

Package and deploy the Streamlit app for easy access, either as a web app or a standalone desktop app.

---

## Author

**Akhil V Nair** – HackP 2025