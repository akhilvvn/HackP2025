import os
import csv
import sys
import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim
import cv2
import streamlit as st
import pandas as pd

original_dir = "samples/original"
altered_dir = "samples/altered"
output_dir = "outputs"
orb_dir = os.path.join(output_dir, "orb_matches")
os.makedirs(output_dir, exist_ok=True)
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

def generate_altered_image(image, rotation_angle=0, resize_factor=100, grayscale=False, hflip=False, vflip=False):
    img = image.copy()
    if rotation_angle != 0:
        img = img.rotate(rotation_angle)
    if resize_factor != 100:
        img = img.resize((int(img.width*resize_factor/100), int(img.height*resize_factor/100)))
    if grayscale:
        img = img.convert("L").convert("RGB")
    if hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def run_cli():
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
        best_results.append(best_row)
    with open(os.path.join(output_dir, "scores_full.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Original", "Altered", "pHash", "dHash", "SSIM", "ORB"])
        writer.writerows(all_results)
    with open(os.path.join(output_dir, "scores.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Original", "Altered", "pHash", "dHash", "SSIM", "ORB"])
        writer.writerows(best_results)
    print(f"Results saved to {output_dir}/scores.csv (best), scores_full.csv (all), and orb_matches/")

def colored_score(score):
    if score < 0.5:
        return f"<span style='color:red'>{score:.2f}</span>"
    return f"{score:.2f}"

def summarize_failures(scores, threshold=0.5):
    summary = {}
    for method in ["pHash","dHash","SSIM","ORB"]:
        low = [s["Altered"] for s in scores if s[method]<threshold]
        high = [s["Altered"] for s in scores if s[method]>=threshold]
        summary[method] = {"fail": low, "strong": high}
    return summary

def run_streamlit_ui():
    st.title("Image Similarity Scoring")

    if "results_cache" not in st.session_state:
        st.session_state.results_cache = []

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    rotation_slider = st.slider("Rotation angle (°)", 0, 360)
    resize_slider = st.slider("Resize (%)", 10, 200, 100)
    grayscale_checkbox = st.checkbox("Grayscale")
    hflip_checkbox = st.checkbox("Horizontal flip")
    vflip_checkbox = st.checkbox("Vertical flip")

    if uploaded_file:
        original = Image.open(uploaded_file).convert("RGB")

        altered = generate_altered_image(
            original,
            rotation_angle=rotation_slider,
            resize_factor=resize_slider,
            grayscale=grayscale_checkbox,
            hflip=hflip_checkbox,
            vflip=vflip_checkbox
        )

        phash_score = hash_similarity(imagehash.phash(original), imagehash.phash(altered))
        dhash_score = hash_similarity(imagehash.dhash(original), imagehash.dhash(altered))
        gray_o = to_gray_array(original)
        gray_a = to_gray_array(altered)
        h = min(gray_o.shape[0], gray_a.shape[0])
        w = min(gray_o.shape[1], gray_a.shape[1])
        ssim_score = ssim(cv2.resize(gray_o, (w,h)), cv2.resize(gray_a, (w,h)))

        def orb_similarity_with_vis(img1, img2):
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(to_gray_array(img1), None)
            kp2, des2 = orb.detectAndCompute(to_gray_array(img2), None)
            if des1 is None or des2 is None or len(kp1)==0 or len(kp2)==0:
                return 0.0, np.hstack([np.array(img1), np.array(img2)])
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if not matches:
                return 0.0, np.hstack([np.array(img1), np.array(img2)])
            good_matches = [m for m in matches if m.distance < 64]
            score = len(good_matches)/len(matches)
            match_img = cv2.drawMatches(np.array(img1), kp1, np.array(img2), kp2, good_matches[:20], None, flags=2)
            return score, match_img

        orb_score, orb_vis = orb_similarity_with_vis(original, altered)

        alter_desc = f"Rotation:{rotation_slider}, Resize:{resize_slider}"
        if grayscale_checkbox: alter_desc += ", Gray"
        if hflip_checkbox: alter_desc += ", HFlip"
        if vflip_checkbox: alter_desc += ", VFlip"

        st.session_state.results_cache.append({
            "Altered": alter_desc,
            "pHash": phash_score,
            "dHash": dhash_score,
            "SSIM": ssim_score,
            "ORB": orb_score
        })

        st.subheader("Side by Side Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original", use_container_width=True)
        with col2:
            st.image(altered, caption="Altered", use_container_width=True)
            st.markdown(f"**pHash:** {colored_score(phash_score)}", unsafe_allow_html=True)
            st.markdown(f"**dHash:** {colored_score(dhash_score)}", unsafe_allow_html=True)
            st.markdown(f"**SSIM:** {colored_score(ssim_score)}", unsafe_allow_html=True)
            st.markdown(f"**ORB:** {colored_score(orb_score)}", unsafe_allow_html=True)

        st.image(Image.fromarray(orb_vis), caption="ORB Match", use_container_width=True)

        st.subheader("Test Where They Fail / Strengths")
        summary = summarize_failures(st.session_state.results_cache)
        for method, info in summary.items():
            st.markdown(
                f"**{method}**: Fail - {', '.join(info['fail']) if info['fail'] else 'None'}; "
                f"Strong - {', '.join(info['strong']) if info['strong'] else 'None'}"
            )

        df = pd.DataFrame(st.session_state.results_cache)
        csv = df.to_csv(index=False)
        st.download_button(
            "Download similarity scores as CSV",
            data=csv,
            file_name="similarity_scores.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit_ui()
    else:
        run_cli()
