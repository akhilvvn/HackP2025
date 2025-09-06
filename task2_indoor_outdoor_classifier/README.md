# Indoor/Outdoor Image Classifier

## Task Description

This project classifies images into five categories: **hotel, indoor, outdoor, park, and restaurant** using deep learning. The classifier uses **ResNet18** and is designed to work well even on small datasets. Key features:

- **Layer4 + fully connected layer fine-tuning**
- **Weighted loss to handle class imbalance**
- **Data augmentation** (random flips, rotations, crops, color jitter)
- **Early stopping** to prevent overfitting
- **Best model saving** for reliable inference
- **Streamlit-based UI** for single image and batch test evaluation

The goal is to achieve high per-class accuracy, provide confusion matrices, and allow easy predictions on new images.

---

## Project Structure

task2_indoor_outdoor_classifier/
│── README.md
│── requirements.txt
│── classifier.py
│── samples/
│   ├── test/
│   └── train/
│── outputs/
    ├── predictions.csv        # Test data predictions
    └── plots/                 # Confusion matrix

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

**For training the model**: Trains the model with fine-tuning, weighted loss, and oversampling. Saves the best model automatically.

```

python classifier.py --mode train

```

**Run CLI inference**: Loads the best saved model. Predicts classes for test images and saves outputs and confusion matrix.

```

python classifier.py --mode cli

```

**Launch Streamlit UI**: Provides a user-friendly interface for training, single image classification, and batch evaluation.

```

streamlit run classifier.py

```


### Outputs will be saved in the outputs/ folder:

predictions.csv → Predicted class for each test image

plots/ → Confusion matrix plots

---

## Sample Results

From `predictions.csv`:

| Image                 | PredictedClass |
|-----------------------|----------------|
| hotel1.jpg            | hotel          |
| hotel2.jpg            | hotel          |
| hotel3.jpg            | hotel          |
| hospital1.jpg         | indoor         |
| library1.jpg          | indoor         |
| ward1.jpg             | indoor         |
| highway1.jpg          | outdoor        |
| scenery1.jpg          | outdoor        |
| street1.webp          | outdoor        |
| park1.jpg             | park           |

---

### Observations

**All classes (hotel, indoor, outdoor, park, restaurant) achieved 100% accuracy on the current test set**

**Weighted loss and oversampling helped balance minority classes**

**Layer4 fine-tuning enabled the pretrained ResNet18 to adapt effectively to the small dataset**

**Generic augmentation improved generalization without adding complexity**

**Early stopping prevented overfitting**

**Streamlit UI enables interactive classification and evaluation**

**While the current results are perfect, a larger and more diverse dataset is needed to ensure robust performance, avoid confusion for visually similar images, and improve generalization on unseen images**

---

## Possible Extensions

Data: Increase dataset size → more robust and generalizable.

Model: Gradually unfreeze layer3 → finer feature learning.

Advanced learning methods: Few-shot or metric learning → handle very small datasets efficiently.

UI Enhancements: Improve the Streamlit interface → better visualization of predictions, confidence scores, and evaluation metrics.


## Author

**Akhil V Nair** – HackP 2025