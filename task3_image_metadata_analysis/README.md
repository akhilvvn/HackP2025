# Image Metadata Analysis

## Task Description

This project extracts and analyzes metadata from images, including EXIF/IPTC fields, GPS location, OCR-based text extraction, and language detection. It provides structured outputs in CSV/JSON formats and generates an interactive HTML report with metadata tables, OCR results, and clickable Google Maps links for images containing GPS coordinates. Additional implemented:

- **OCR text extraction with Tesseract**
- **Automatic language detection of extracted text**
- **Clickable Google Maps links for GPS metadata**
- **Visualization of image properties and EXIF presence**

The goal is to explore what information can be pulled from an image’s metadata and infer extra context (such as hidden camera details, text content, language, and location) to assist in digital forensics and authenticity verification.

---

## Project Structure

task3_image_metadata_analysis/
│── README.md
│── requirements.txt
│── metadata_extractor.py
│── samples/
│   └── images/              
│── outputs/
    ├── metadata.csv          
    ├── metadata.json         
    ├── report.html           
    └── plots/                
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

⚠️ Additional setup for OCR:

Install Tesseract OCR on your system:

On Windows Installer: https://github.com/UB-Mannheim/tesseract/wiki

Ensure it is available in your PATH or update the script with the correct Tesseract path if needed, like so:

```
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

```

**Linux**

```

sudo apt install tesseract-ocr

```

**Mac**

```

brew install tesseract

```

---

## Usage

Run the extractor on all images inside samples/images/:
```

python metadata_extractor.py

```

### Outputs will be saved in the outputs/ folder:

metadata.csv → Extracted metadata in CSV format

metadata.json → Extracted metadata in JSON format

report.html → Interactive HTML report with plots and metadata table

plots/ → Visualization of image statistics

---

## Sample Results

From `metadata.csv`:

| filename          | format | width | height | filesize\_bytes | ocr\_text                             | ocr\_language | Image Make | Image Model  | GPSLink                                                                                        |
| ----------------- | ------ | ----- | ------ | --------------- | ------------------------------------- | ------------- | ---------- | ------------ | ---------------------------------------------------------------------------------------------- |
| sign3.jpg         | JPEG   | 1300  | 1064   | 272698          | [www.alamy.com](http://www.alamy.com) | en            |            |              |                                                                                                |
| screenshot1.png   | PNG    | 1902  | 1016   | 1157171         | "... Konsole ..."                     | en            |            |              |                                                                                                |
| photoshopped3.jpg | JPEG   | 780   | 518    | 453192          |                                       |               | Canon      | Canon MG3500 |                                                                                                |
| mobile1.jpg       | JPEG   | 768   | 361    | 39954           |                                       |               | Samsung    | Galaxy S21   | [https://www.google.com/maps?q=12.9716,77.5946](https://www.google.com/maps?q=12.9716,77.5946) |

Plots generated:

Image format distribution

File size distribution (KB)

Top image resolutions

EXIF presence ratio

---

### Observations

**Smartphone photos: rich metadata (camera make, model, GPS)**

**Edited photos: show Software fields (e.g., Photoshop)**

**Screenshots: lack EXIF, but OCR extracts UI text**

**Stock images: often stripped of metadata**

**Text-rich images: OCR + language detection enhance context understanding**

Some images carried detailed EXIF data (camera model, software, timestamp), while others had stripped metadata (e.g., screenshots, edited images).

OCR worked effectively on images with visible text, and language detection correctly identified English and other languages.

GPS coordinates (if present) were extracted and linked to Google Maps for easy verification.

Metadata absence in some images (e.g., PNG, screenshots) confirms common practice of stripping EXIF during editing.

---

## Possible Extensions

Deeper Metadata Parsing: Add IPTC, XMP, and ICC profile extraction for richer forensic insights.

Tampering Detection: Detect editing/manipulation by analyzing inconsistencies in EXIF or compression signatures.

Reverse Geocoding: Convert GPS coordinates into actual place names using a geocoding API.

EXIF Timeline Analysis: Reconstruct chronological timelines across multiple images.

Web/Cloud Deployment: Package as a Streamlit dashboard or lightweight web API for practical forensic use.

---

## Author

**Akhil V Nair** – HackP 2025