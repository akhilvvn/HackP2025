import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

parser = argparse.ArgumentParser(description="Train or run inference on the classifier.")
parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
args = parser.parse_args()
MODE = args.mode

train_dir = "samples/train"
test_dir = "samples/test"
output_dir = "outputs"
plots_dir = os.path.join(output_dir, "plots")
BEST_MODEL_PATH = os.path.join(output_dir, "best_model.pth")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if MODE == "train":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

class_counts = [0] * len(train_dataset.classes)
for _, label in train_dataset.samples:
    class_counts[label] += 1
weights_per_class = [sum(class_counts)/c for c in class_counts]
samples_weight = [weights_per_class[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Detected classes: {class_names}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

class_weights = torch.tensor(weights_per_class, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 5e-4}
])

if MODE == "train":
    best_accuracy = 0.0
    epochs = 10
    patience = 3
    no_improve = 0
    print("Starting training...")

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

if MODE == "inference":
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {BEST_MODEL_PATH}")

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model = model.to(device)
    model.eval()
    print("Loaded best model for inference.")

    y_true, y_pred = [], []
    predictions = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            batch_start = batch_idx * test_loader.batch_size
            batch_paths = [test_loader.dataset.samples[j][0]
                           for j in range(batch_start, batch_start + len(inputs))]

            for path, pred_idx in zip(batch_paths, preds.cpu().numpy()):
                predictions.append([os.path.basename(path), class_names[pred_idx]])

    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "PredictedClass"])
        writer.writerows(predictions)
    print(f"Predictions saved to {csv_path}")

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

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
