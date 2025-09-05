# Image Metadata Analysis

## Task Description

This project extracts and analyzes metadata from images, including EXIF/IPTC fields, GPS location, OCR-based text extraction, and language detection. It provides structured outputs in CSV/JSON formats and generates an interactive HTML report with metadata tables, OCR results, and clickable Google Maps links for images containing GPS coordinates. Additional features implemented:

- **OCR text extraction with Tesseract**
- **Automatic language detection of extracted text**
- **Clickable Google Maps links for GPS metadata**
- **Visualization of image properties and EXIF presence**
- **Interactive GUI for real-time analysis**
- **Downloadable metadata in CSV and JSON formats**

The goal is to explore what information can be pulled from an image’s metadata and infer extra context (such as hidden camera details, text content, language, and location) to assist in digital forensics and authenticity verification.

---

## Project Structure

task3_image_metadata_analysis/
│── README.md
│── requirements.txt
│── metadata_extractor.py
│── samples/
│   └── images/
             
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

Run the Streamlit app:

```

streamlit run metadata_extractor.py

```

---

## Outputs

Interactive metadata table displayed in the app

Download options: CSV and JSON formats for all extracted metadata

Inline visualizations: image format distribution, file size distribution, top resolutions, and EXIF presence

Missing data is displayed as Not available for clarity.

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

Deeper Metadata Parsing: Extend support to extract IPTC, XMP, and ICC profile data for richer forensic insights.

Tampering Detection: Analyze inconsistencies in EXIF or compression signatures to detect edited or manipulated images.

Reverse Geocoding: Convert GPS coordinates into human-readable locations (city, country) using geocoding APIs.

EXIF Timeline Analysis: Reconstruct chronological timelines across multiple images to track event sequences.

Enhanced OCR & Context Analysis: Support multi-language detection, handwriting recognition, or entity extraction from text within images.

Batch Analysis & Reporting: Enable bulk image uploads with interactive filtering, sorting, and downloadable summary reports.

---

## Author

**Akhil V Nair** – HackP 2025