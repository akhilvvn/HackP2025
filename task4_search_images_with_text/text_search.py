import os
import shutil
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np

dataset_dir = "samples/dataset"
output_dir = "outputs"
preview_dir = os.path.join(output_dir, "previews")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(preview_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

print(f"Found {len(image_paths)} images in dataset.")

image_embeddings = []
for path in image_paths:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    image_embeddings.append(emb.cpu().numpy())

image_embeddings = np.vstack(image_embeddings)
print("Image embeddings computed and cached.")

n_clusters = min(10, len(image_paths))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(image_embeddings)
print(f"Images clustered into {n_clusters} clusters.")

similarity_threshold = 0.25
max_results_per_query = 2
results = []

while True:
    query = input("\nEnter search query (or type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        break
    if not query:
        continue

    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.cpu().numpy()

    sims = cosine_similarity(text_emb, image_embeddings)[0]

    cluster_sims = cosine_similarity(text_emb, kmeans.cluster_centers_)
    best_cluster = cluster_sims.argmax()
    cluster_indices = np.where(cluster_labels == best_cluster)[0]

    valid_indices = [i for i in cluster_indices if sims[i] >= similarity_threshold]
    top_indices = valid_indices[:max_results_per_query]

    query_preview_dir = os.path.join(preview_dir, query.replace(" ", "_"))
    os.makedirs(query_preview_dir, exist_ok=True)

    for rank, idx in enumerate(top_indices, start=1):
        src = image_paths[idx]
        dst = os.path.join(query_preview_dir, f"rank{rank}_{os.path.basename(src)}")
        shutil.copy(src, dst)
        results.append({
            "query": query,
            "rank": rank,
            "filename": os.path.basename(src),
            "similarity": float(sims[idx])
        })

    if top_indices:
        print(f"Top-{len(top_indices)} results saved in {query_preview_dir}")
    else:
        print("No images above similarity threshold found for this query.")

df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "results.csv")
df.to_csv(csv_path, index=False)
print(f"\nAll results saved to {csv_path}")
