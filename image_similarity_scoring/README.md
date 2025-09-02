# Image Similarity Scoring

## Task Description

This task compares an original image with its altered versions (rotated, resized, blurred, filtered, etc.) and computes similarity scores using multiple methods:

- **pHash (Perceptual Hash)**
- **dHash (Difference Hash)**
- **SSIM (Structural Similarity Index)**
- **ORB Feature Matching (OpenCV)**

The goal is to evaluate how well different similarity measures perform under common distortions.

---

## Project Structure

image_similarity_scoring/
│── README.md
│── requirements.txt
│── similarity_scoring.py
│── samples/
│   ├── original/
│   └── altered/
│── outputs/
    ├── scores.csv        # Best-match
    ├── scores_full.csv   # All comparisons
    ├── plots/
    └── orb_matches/

---

## Installation

It is recommended to use a **virtual environment**.

### Create and activate venv

**Linux / Mac**
python3 -m venv venv
source venv/bin/activate

**On Windows**
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

---

## Usage

Run the similarity scoring script:

python similarity_scoring.py

### Outputs will be saved in the outputs/ folder:

scores.csv → Best match per altered image

scores_full.csv → All original vs altered comparisons

plots/ → Side-by-side image comparisons with scores

orb_matches/ → ORB keypoint match diagrams

---

## Sample Results

From `scores.csv`:

| Original   | Altered             | pHash  | dHash   | SSIM  | ORB   |
|------------|---------------------|--------|---------|-------|-------|
| board.jpg  | board-b&w.jpg       | 1.0000 | 0.9531  | 0.9603 | 0.9945 |
| cat.jpg    | cat-1degree.jpg     | 0.8750 | 0.9688  | 0.3586 | 0.9758 |
| cat.jpg    | cat-blurred.jpg     | 0.9688 | 1.0000  | 0.4857 | 0.6667 |
| dog.jpg    | dog-10degrees.jpg   | 0.6250 | 0.8125  | 0.4781 | 0.7475 |
| girlshop.jpg | girlshop-17degrees.jpg | 0.6563 | 0.7031 | 0.3323 | 0.9598 |
| girlshop.jpg | girlshop-resized.jpg   | 0.5625 | 0.6250 | 0.2310 | 0.9307 |
| tree.jpg   | tree-contrasted.jpg | 0.9375 | 0.9531  | 0.9123 | 0.9637 |

---

### Observations
- **Hashes (pHash, dHash)** → detect grayscale/contrast changes strongly (`board-b&w.jpg` scores near 1.0).  
- **SSIM** → weaker on rotations but useful for compression and mild distortions.  
- **ORB** → robust against rotations and resizing (`girlshop-17degrees.jpg`, `girlshop-resized.jpg` show strong ORB matches).

Different similarity methods have different strengths.

Combining them yields stronger, more reliable matching.

ORB provides robustness where pixel-based methods fail.

---

## Possible Extensions

CLIP embeddings for semantic similarity.

Heatmap visualization of similarity matrices.

Integration with FAISS for large-scale image search.


## Author

**Akhil V Nair** – HackP 2025

