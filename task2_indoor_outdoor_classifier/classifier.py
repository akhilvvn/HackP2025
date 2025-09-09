import os
import csv
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Train, run CLI inference, or launch UI.")
parser.add_argument('--mode', type=str, choices=['train', 'cli', 'ui'], default=None)
args, unknown = parser.parse_known_args()

MODE = args.mode
train_dir = "samples/train"
test_dir = "samples/test"
output_dir = "outputs"
plots_dir = os.path.join(output_dir, "plots")
BEST_MODEL_PATH = os.path.join(output_dir, "best_model.pth")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def prepare_datasets():
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    weights_per_class = [sum(class_counts)/c for c in class_counts]
    samples_weight = [weights_per_class[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader, weights_per_class

def create_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

def train_model():
    train_dataset, test_dataset, train_loader, test_loader, weights_per_class = prepare_datasets()
    class_names = train_dataset.classes
    num_classes = len(class_names)
    model = create_model(num_classes)
    class_weights = torch.tensor(weights_per_class, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 5e-4}
    ])
    best_accuracy = 0.0
    epochs = 10
    patience = 3
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        epoch_accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.4f}")
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved new best model with accuracy {best_accuracy:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping: no improvement for {patience} epochs")
                break
    print(f"Training complete. Best accuracy: {best_accuracy:.4f}")

def run_cli():
    train_dataset, test_dataset, train_loader, test_loader, weights_per_class = prepare_datasets()
    class_names = train_dataset.classes
    num_classes = len(class_names)
    model = create_model(num_classes)
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    y_true, y_pred, predictions = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            batch_start = batch_idx * test_loader.batch_size
            batch_paths = [test_loader.dataset.samples[j][0] for j in range(batch_start, batch_start + len(inputs))]
            for path, pred_idx in zip(batch_paths, preds.cpu().numpy()):
                predictions.append([os.path.basename(path), class_names[pred_idx]])
    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "PredictedClass"])
        writer.writerows(predictions)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for t, p in zip(y_true, y_pred):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1
    print("Per-class accuracy:")
    for i, cname in enumerate(class_names):
        acc = (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
        print(f"{cname}: {class_correct[i]}/{class_total[i]} = {acc:.2f}%")
    overall_acc = sum(class_correct)/sum(class_total)*100 if sum(class_total) > 0 else 0
    print(f"Overall accuracy: {overall_acc:.2f}%")

def run_ui():
    train_dataset, test_dataset, train_loader, test_loader, weights_per_class = prepare_datasets()
    class_names = train_dataset.classes
    num_classes = len(class_names)
    model = create_model(num_classes)

    st.title("Indoor/Outdoor Image Classifier")
    st.sidebar.header("Options")

    if st.sidebar.button("Train Model"):
        train_model()
        st.success("Training complete!")

    if not os.path.exists(BEST_MODEL_PATH):
        st.warning("Best model not found. Train the model first to classify images.")
        st.stop()

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    app_mode = st.sidebar.radio("Select Mode", ["Single Image", "Batch Test Set"])
    show_accuracy = st.sidebar.checkbox("Show Per-Class Accuracy", value=True)

    if app_mode == "Single Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            input_tensor = transform_test(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            topk = np.argsort(probs)[::-1][:3]
            st.subheader("Predictions")
            for i in topk:
                st.write(f"{class_names[i]}: {probs[i]*100:.2f}%")
            st.bar_chart(pd.DataFrame({"Confidence": probs}, index=class_names))

    elif app_mode == "Batch Test Set":
        if st.button("Run Evaluation"):
            y_true, y_pred, predictions = [], [], []
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(test_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    batch_start = batch_idx * test_loader.batch_size
                    batch_paths = [test_loader.dataset.samples[j][0] for j in range(batch_start, batch_start + len(inputs))]
                    for path, pred_idx in zip(batch_paths, preds.cpu().numpy()):
                        predictions.append([os.path.basename(path), class_names[pred_idx]])

            cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
            st.pyplot(fig)

            df_preds = pd.DataFrame(predictions, columns=["Image", "PredictedClass"])
            csv = df_preds.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

            if show_accuracy:
                class_correct = [0]*num_classes
                class_total = [0]*num_classes
                for t, p in zip(y_true, y_pred):
                    class_total[t] += 1
                    if t == p:
                        class_correct[t] += 1
                acc_data = {"Class": [], "Correct": [], "Total": [], "Accuracy %": []}
                for i, cname in enumerate(class_names):
                    acc = (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
                    acc_data["Class"].append(cname)
                    acc_data["Correct"].append(class_correct[i])
                    acc_data["Total"].append(class_total[i])
                    acc_data["Accuracy %"].append(round(acc, 2))
                df_acc = pd.DataFrame(acc_data)
                st.subheader("Per-Class Accuracy")
                st.dataframe(df_acc)
                overall_acc = sum(class_correct)/sum(class_total)*100 if sum(class_total) > 0 else 0
                st.write(f"**Overall Accuracy:** {overall_acc:.2f}%")

if __name__ == "__main__":
    if MODE == "train":
        train_model()
    elif MODE == "cli":
        run_cli()
    else:
        run_ui()
