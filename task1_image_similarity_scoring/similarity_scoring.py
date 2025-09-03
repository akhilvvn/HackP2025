import os
import csv
import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt

original_dir = "samples/original"
altered_dir = "samples/altered"
output_dir = "outputs"
plots_dir = os.path.join(output_dir, "plots")
orb_dir = os.path.join(output_dir, "orb_matches")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(orb_dir, exist_ok=True)

def load_image(path):
    return Image.open(path).convert("RGB")

def to_gray_array(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

def hash_similarity(hash1, hash2):
    return 1 - (hash1 - hash2) / (len(hash1.hash) ** 2)

def orb_similarity(img1, img2, out_path=None):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(to_gray_array(img1), None)
    kp2, des2 = orb.detectAndCompute(to_gray_array(img2), None)
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0
    good_matches = [m for m in matches if m.distance < 64]
    score = len(good_matches) / len(matches)
    if out_path:
        match_img = cv2.drawMatches(np.array(img1), kp1, np.array(img2), kp2, matches[:20], None, flags=2)
        cv2.imwrite(out_path, match_img)
    return score

# Precompute originals
originals = {}
for original_file in os.listdir(original_dir):
    path = os.path.join(original_dir, original_file)
    img = load_image(path)
    originals[original_file] = {
        "image": img,
        "phash": imagehash.phash(img),
        "dhash": imagehash.dhash(img),
        "gray": to_gray_array(img),
    }

all_results = []
best_results = []

for altered_file in os.listdir(altered_dir):
    altered_path = os.path.join(altered_dir, altered_file)
    img_altered = load_image(altered_path)
    phash_alt = imagehash.phash(img_altered)
    dhash_alt = imagehash.dhash(img_altered)
    gray_alt = to_gray_array(img_altered)

    best_score = -1
    best_row = None
    best_original = None

    for original_file, data in originals.items():
        phash_score = hash_similarity(data["phash"], phash_alt)
        dhash_score = hash_similarity(data["dhash"], dhash_alt)

        h = min(data["gray"].shape[0], gray_alt.shape[0])
        w = min(data["gray"].shape[1], gray_alt.shape[1])
        gray_o = cv2.resize(data["gray"], (w, h))
        gray_a = cv2.resize(gray_alt, (w, h))
        ssim_score = ssim(gray_o, gray_a)

        orb_path = os.path.join(orb_dir, f"{original_file}_{altered_file}.jpg")
        orb_score = orb_similarity(data["image"], img_altered, out_path=orb_path)

        row = [original_file, altered_file, phash_score, dhash_score, ssim_score, orb_score]
        all_results.append(row)

        if phash_score + dhash_score + ssim_score + orb_score > best_score:
            best_score = phash_score + dhash_score + ssim_score + orb_score
            best_row = row
            best_original = data["image"]

    best_results.append(best_row)

    if best_original is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(best_original)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(img_altered)
        axes[1].set_title(
            f"Altered\npHash={best_row[2]:.2f}, dHash={best_row[3]:.2f}, SSIM={best_row[4]:.2f}, ORB={best_row[5]:.2f}"
        )
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{best_row[0]}_{altered_file}.png"))
        plt.close()

with open(os.path.join(output_dir, "scores_full.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Original", "Altered", "pHash", "dHash", "SSIM", "ORB"])
    writer.writerows(all_results)

with open(os.path.join(output_dir, "scores.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Original", "Altered", "pHash", "dHash", "SSIM", "ORB"])
    writer.writerows(best_results)

print(f"Results saved to {output_dir}/scores.csv (best), scores_full.csv (all), plots/, and orb_matches/")
