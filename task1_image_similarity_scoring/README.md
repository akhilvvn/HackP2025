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

task1_image_similarity_scoring/
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

| Original     | Altered                | pHash | dHash | SSIM | ORB  |
| ------------ | ---------------------- | ----- | ----- | ---- | ---- |
| girlshop.jpg | girlshop-1degree.jpg   | 0.94  | 0.94  | 0.54 | 1.00 |
| girlshop.jpg | girlshop-10degree.jpg  | 0.72  | 0.81  | 0.37 | 0.99 |
| girlshop.jpg | girlshop-17degrees.jpg | 0.66  | 0.70  | 0.33 | 0.96 |
| girlshop.jpg | girlshop-b\&w\.jpg     | 0.97  | 0.97  | 0.95 | 0.97 |
| girlshop.jpg | girlshop-resized.jpg   | 0.56  | 0.63  | 0.23 | 0.93 |
| girlshop.jpg | girlshop-rotated.jpg   | 0.53  | 0.50  | 0.33 | 1.00 |
| girlshop.jpg | girlshop-vintage.jpg   | 0.88  | 0.91  | 0.81 | 0.99 |
| tree.jpg     | tree-contrasted.jpg    | 0.94  | 0.95  | 0.91 | 0.96 |
| board.jpg    | board-b\&w\.jpg        | 1.00  | 0.95  | 0.96 | 0.99 |
| cat.jpg      | cat-blurred.jpg        | 0.97  | 1.00  | 0.49 | 0.67 |

---

### Observations
- **Hashes (pHash, dHash)** → Very robust against grayscale and contrast changes (board.jpg vs board-b&w.jpg both ≈0.95–1.0). Capture overall structure but less sensitive to blur.
- **SSIM** → Performs well under mild compression or brightness change. Fails under rotations: drops heavily even at small angles (1°, 10°, 17°).  
- **ORB** → Robust to rotations, blur, and small geometric changes. Even at 17° rotation, ORB similarity stays high (≈0.96) where SSIM collapses. Fails if not enough keypoints are detected (heavily blurred or low-texture regions).

Rotation Sensitivity:

At 1° rotation:

Hashes ≈0.87–0.97

SSIM drops to ≈0.35

ORB stays strong (≈0.97)

At 10° rotation:

Hashes ≈0.63–0.81

SSIM collapses further (<0.25 for some images).

ORB remains reliable (≈0.9–0.99).

At 17° rotation:

Hashes ≈0.65–0.70

SSIM drops to ≈0.33

ORB stays strong (≈0.95–0.96)

Blur

SSIM and hashes partially degrade.

ORB can fail completely if keypoints vanish.

Resizing

SSIM penalizes heavily, but ORB still detects similarity.

Filters

pHash/dHash stay high, SSIM also remains strong.

Different similarity methods have different strengths.

Combining them yields stronger, more reliable matching.

ORB provides robustness where pixel-based methods fail, especially in rotation-heavy cases.

SSIM is useful for compression/contrast detection but not geometric distortions.

---

## Possible Extensions

CLIP embeddings for semantic similarity.

Heatmap visualization of similarity matrices.

Integration with FAISS for large-scale image search.


## Author

**Akhil V Nair** – HackP 2025