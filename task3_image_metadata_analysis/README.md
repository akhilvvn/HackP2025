# Image Metadata Analysis

## Task Description

This project extracts, analyzes, and visualizes metadata from images to support **digital forensics** and **authenticity verification**. It goes beyond basic EXIF parsing by incorporating:

- **Device details** (make, model, lens, software)
- **Camera settings** (exposure, ISO, focal length, flash, white balance)
- **Timestamps** (original, digitized, modified)
- **GPS coordinates** with clickable Google Maps links
- **OCR text extraction** using Tesseract
- **Automatic language detection** of OCR results
- **File properties** (size, resolution, format)
- **Anomaly checks** (suspicious timestamps, editing software traces, missing EXIF)
- **AI-generated detection heuristics** (flags StableDiffusion, Midjourney, DALL·E, etc.)
- **Interactive visualizations** (format distribution, file size histogram, top resolutions, EXIF presence)
- **Downloadable structured outputs** in CSV and JSON

The goal is to uncover all information embedded in an image’s metadata and infer additional context—such as hidden camera details, timestamps, text content, language, location, editing traces, and potential AI-generation—to provide rapid insights for digital forensics and authenticity verification.

---

## Project Structure

HackP2025/
│── task3_image_metadata_analysis/
│   │── README.md
│   │── requirements.txt
│   │── metadata_extractor.py
│   │
│   └── samples/
│       └── images/
             
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

## Install dependencies

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

Metadata Table with device info, timestamps, GPS, OCR results, anomaly analysis, AI-generated detection.

Downloadable CSV/JSON containing all extracted metadata

Visualizations: Image format distribution, File size histogram (KB), Top resolutions, EXIF presence ratio

Inline visualizations: image format distribution, file size distribution, top resolutions, and EXIF presence

Subtle anomaly insights highlighting: Edited/modified files, Suspicious timestamps, Missing EXIF/GPS data, Potential AI-generated images

---

## Sample Results

| filename                          | format | width | height | filesize_bytes | device_make | device_model        | software           | datetime_original     | GPSLink                                                                                      | ocr_text                                              | ocr_language | analysis_summary                        | ai_generated          |
|----------------------------------|--------|-------|--------|----------------|-------------|--------------------|------------------|---------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------|--------------|----------------------------------------|----------------------|
| DSCN0010.jpg                      | JPEG   | 640   | 480    | 161713         | NIKON       | COOLPIX P6000      | Nikon Transfer 1.1 W | 2008:10:22 16:28:39 | [43.467448,11.885127](https://www.google.com/maps?q=43.46744833333334,11.885126666663888) | Not available                                       | Not available | No anomalies detected                   | No indication        |
| canon-ixus.jpg                     | JPEG   | 640   | 480    | 128037         | Canon       | Canon DIGITAL IXUS | Not available     | 2001:06:09 15:17:32 | Not available                                                                                | Not available                                       | Not available | Location data missing                   | No indication        |
| eurotext.png                       | PNG    | 1024  | 800    | 14756          | Not available | Not available     | Not available     | Not available        | Not available                                                                                | "The (quick) [brown] {fox} jumps! Over the $43,456..." | la (-971.12) | Device information missing | Location data missing | No indication        |
| IMG_20250909_231221007.jpg        | JPEG   | 4080  | 3072   | 3039288        | Nothing     | A059               | Not available     | 2025:09:09 23:12:21 | [8.497636,76.999053](https://www.google.com/maps?q=8.49763611111111,76.99905277777778)    | "ES oC el? RORY, STS v i"                           | en (9.06)    | No anomalies detected                   | No indication        |
| PXL_20250509_170641209.PORTRAIT.ORIGINAL.jpg | JPEG | 3072  | 4080   | 1355149        | moto g45 5G (fogos) | motorola        | HDR+ 1.0.604778939zp | 2025:05:09 17:06:41 | Not available                                                                                | Not available                                       | Not available | Location data missing                   | No indication        |

---

### Observations

**Smartphone photos: rich metadata (camera make, model, timestamps GPS)**

**Edited photos: flagged by software traces and inconsistent EXIF**

**Screenshots: lack EXIF, but OCR extracts UI text**

**Stock/Compressed images: often stripped of metadata**

**Text-rich images: OCR + language detection enhance context understanding**

**AI-generated images → often lack device details but include generator signatures**

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