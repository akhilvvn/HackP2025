import os
import torch
import pickle
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATASET_DIR = "samples/dataset"
OUTPUT_DIR = "outputs"
OBJECTS_DIR = os.path.join(OUTPUT_DIR, "dataset_objects")
DEFAULT_THRESHOLD = 0.4

os.makedirs(OBJECTS_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = YOLO("yolo11x.pt")
clip_model = SentenceTransformer("clip-ViT-B-32")

dataset_embeddings_file = os.path.join(OBJECTS_DIR, "embeddings.pt")
dataset_paths_file = os.path.join(OBJECTS_DIR, "paths.pkl")
dataset_conf_file = os.path.join(OBJECTS_DIR, "confidences.pkl")
dataset_labels_file = os.path.join(OBJECTS_DIR, "labels.pkl")

if (
    os.path.exists(dataset_embeddings_file)
    and os.path.exists(dataset_paths_file)
    and os.path.exists(dataset_conf_file)
    and os.path.exists(dataset_labels_file)
):
    dataset_embeddings = torch.load(dataset_embeddings_file)
    with open(dataset_paths_file, "rb") as f:
        dataset_object_paths = pickle.load(f)
    with open(dataset_conf_file, "rb") as f:
        dataset_object_confs = pickle.load(f)
    with open(dataset_labels_file, "rb") as f:
        dataset_object_labels = pickle.load(f)
else:
    dataset_embeddings = []
    dataset_object_paths = []
    dataset_object_confs = []
    dataset_object_labels = []
    for img_name in tqdm(os.listdir(DATASET_DIR)):
        img_path = os.path.join(DATASET_DIR, img_name)
        image = Image.open(img_path).convert("RGB")
        results = yolo_model(np.array(image))
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls in zip(boxes, scores, classes):
                crop = image.crop((box[0], box[1], box[2], box[3]))
                embedding = clip_model.encode(crop, convert_to_tensor=True, device=device)
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
                dataset_embeddings.append(embedding.cpu())
                dataset_object_paths.append(img_path)
                dataset_object_confs.append(float(score))
                dataset_object_labels.append(yolo_model.names[cls])
    dataset_embeddings = torch.vstack(dataset_embeddings)
    torch.save(dataset_embeddings, dataset_embeddings_file)
    with open(dataset_paths_file, "wb") as f:
        pickle.dump(dataset_object_paths, f)
    with open(dataset_conf_file, "wb") as f:
        pickle.dump(dataset_object_confs, f)
    with open(dataset_labels_file, "wb") as f:
        pickle.dump(dataset_object_labels, f)

def detect_query_objects(query_file):
    image = Image.open(query_file).convert("RGB")
    results = yolo_model(np.array(image))
    detected = {}
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls in zip(boxes, scores, classes):
            label = yolo_model.names[cls]
            detected[label] = float(score)
    return detected

def search_query_image(query_file, selected_class=None, threshold=DEFAULT_THRESHOLD):
    image = Image.open(query_file).convert("RGB")
    results = yolo_model(np.array(image), verbose=False)
    query_embeddings = []
    query_confs = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls in zip(boxes, scores, classes):
            label = yolo_model.names[cls]
            if selected_class is not None and label != selected_class:
                continue
            crop = image.crop((box[0], box[1], box[2], box[3]))
            embedding = clip_model.encode(crop, convert_to_tensor=True, device=device)
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
            query_embeddings.append(embedding.cpu())
            query_confs.append(float(score))
    if not query_embeddings:
        return []

    query_embeddings = torch.vstack(query_embeddings)

    if selected_class is not None:
        indices = [i for i, lbl in enumerate(dataset_object_labels) if lbl == selected_class]
        dataset_embeddings_filtered = dataset_embeddings[indices]
        dataset_paths_filtered = [dataset_object_paths[i] for i in indices]
        dataset_confs_filtered = [dataset_object_confs[i] for i in indices]
    else:
        dataset_embeddings_filtered = dataset_embeddings
        dataset_paths_filtered = dataset_object_paths
        dataset_confs_filtered = dataset_object_confs

    sims = query_embeddings @ dataset_embeddings_filtered.T
    sims_np = sims.numpy()
    weighted_sims = sims_np * np.array(query_confs)[:, None] * np.array(dataset_confs_filtered)[None, :]
    max_sims = weighted_sims.max(axis=0)
    sorted_idx = np.argsort(-max_sims)
    top_matches = []
    seen = set()
    for i in sorted_idx:
        img = dataset_paths_filtered[i]
        sim_score = max_sims[i]
        if sim_score < threshold:
            continue
        if img not in seen:
            seen.add(img)
            top_matches.append((img, sim_score))
    return top_matches
