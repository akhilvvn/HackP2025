import os
import torch
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

dataset_dir = "samples/dataset"
embeddings_dir = "embeddings"
os.makedirs(embeddings_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model_and_data():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_paths = sorted([
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])
    emb_file = os.path.join(embeddings_dir, "image_embeddings.npy")
    if os.path.exists(emb_file):
        embeddings = np.load(emb_file)
    else:
        embeddings = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            inputs = {k: v.to(device) for k, v in processor(images=image, return_tensors="pt").items()}
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())
        embeddings = np.vstack(embeddings)
        np.save(emb_file, embeddings)
    cluster_file = os.path.join(embeddings_dir, "clusters.npz")
    if os.path.exists(cluster_file):
        cluster_data = np.load(cluster_file)
        cluster_centers = cluster_data['centers']
        cluster_labels = cluster_data['labels']
    else:
        n_clusters = min(10, len(image_paths))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
        np.savez(cluster_file, centers=cluster_centers, labels=cluster_labels)
    return model, processor, image_paths, embeddings, cluster_centers, cluster_labels

with st.spinner("Loading CLIP model and preparing image dataset... This may take a minute."):
    model, processor, image_paths, image_embeddings, cluster_centers, cluster_labels = load_model_and_data()

st.success("Model and dataset loaded successfully! You can start searching now.")
st.title("Search Images with Text")

query = st.text_input("Enter your search query:")
similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.25, 0.01)
max_results = st.slider("Max results per query", 1, 10, 2)

results = []

if query:
    inputs = {k: v.to(device) for k, v in processor(text=[query], return_tensors="pt", padding=True).items()}
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.cpu().numpy()
    sims = cosine_similarity(text_emb, image_embeddings)[0]
    cluster_sims = cosine_similarity(text_emb, cluster_centers)
    best_cluster = cluster_sims.argmax()
    cluster_indices = np.where(cluster_labels == best_cluster)[0]
    valid_indices = [i for i in cluster_indices if sims[i] >= similarity_threshold]
    top_indices = sorted(valid_indices, key=lambda i: sims[i], reverse=True)[:max_results]
    if top_indices:
        st.subheader(f"Results for: {query}")
        cols = st.columns(len(top_indices))
        for rank, idx in enumerate(top_indices, start=1):
            with cols[rank - 1]:
                st.image(image_paths[idx], caption=f"Rank {rank} | Score {sims[idx]:.3f}", use_container_width=True)
            results.append({
                "query": query,
                "rank": rank,
                "filename": os.path.basename(image_paths[idx]),
                "similarity": float(sims[idx])
            })
    else:
        st.warning("No images above similarity threshold found for this query.")

if results:
    df = pd.DataFrame(results)
    csv_data = df.to_csv(index=False).encode('utf-8')  # Convert to bytes
    st.download_button(
        label="Download results.csv",
        data=csv_data,
        file_name="results.csv",
        mime="text/csv"
    )
