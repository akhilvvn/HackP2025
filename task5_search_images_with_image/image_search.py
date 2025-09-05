import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

dataset_dir = "samples/dataset"
output_dir = "outputs"
preview_dir = os.path.join(output_dir, "previews")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(preview_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

image_embeddings = []
for path in image_paths:
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    image_embeddings.append(emb.cpu().numpy())
import numpy as np
image_embeddings = np.vstack(image_embeddings)

st.title("🖼 Image Search with CLIP")
uploaded_file = st.file_uploader("Upload a query image", type=["jpg","jpeg","png","webp"])

top_k = 2
similarity_threshold = 0.25

if uploaded_file is not None:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Query Image", use_container_width=True)

    inputs = processor(images=query_img, return_tensors="pt").to(device)
    with torch.no_grad():
        query_emb = model.get_image_features(**inputs).cpu().numpy()

    sims = cosine_similarity(query_emb, image_embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    top_indices = [i for i in top_indices if sims[i] >= similarity_threshold]

    if not top_indices:
        st.warning("No images above similarity threshold found.")
    else:
        st.subheader("Top Matches")
        for rank, idx in enumerate(top_indices, start=1):
            matched_img_path = image_paths[idx]
            matched_img = Image.open(matched_img_path).convert("RGB")

            widths, heights = query_img.width + matched_img.width, max(query_img.height, matched_img.height)
            combined = Image.new("RGB", (widths, heights))
            combined.paste(query_img, (0, 0))
            combined.paste(matched_img, (query_img.width, 0))

            st.image(combined, caption=f"Rank {rank} - {os.path.basename(matched_img_path)}", use_container_width=True)

            query_preview_dir = os.path.join(preview_dir, os.path.splitext(os.path.basename(uploaded_file.name))[0])
            os.makedirs(query_preview_dir, exist_ok=True)
            combined.save(os.path.join(query_preview_dir, f"rank{rank}_{os.path.basename(matched_img_path)}"))

        results = [{
            "query": uploaded_file.name,
            "rank": rank,
            "filename": os.path.basename(image_paths[idx]),
            "similarity": float(sims[idx])
        } for rank, idx in enumerate(top_indices, start=1)]

        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "results.csv")
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", index=False, header=False)
        else:
            df.to_csv(csv_path, index=False)
        st.success(f"Results saved to {csv_path}")
