import os
import pandas as pd
from PIL import Image
import exifread
import pytesseract
import matplotlib.pyplot as plt
import langid
import streamlit as st
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Image Metadata Analysis", layout="wide")
st.title("Image Metadata Analysis")

uploaded_files = st.file_uploader("Upload images", type=["jpg","jpeg","png","bmp","tiff"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload one or more images to begin.")
    st.stop()

def extract_gps(tags):
    def conv(values, ref):
        d, m, s = [float(v.num)/float(v.den) for v in values]
        coord = d + m/60.0 + s/3600.0
        if ref in ['S','W']:
            coord *= -1
        return coord
    lat = lon = None
    if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
        lat = conv(tags["GPS GPSLatitude"].values, str(tags["GPS GPSLatitudeRef"]))
        lon = conv(tags["GPS GPSLongitude"].values, str(tags["GPS GPSLongitudeRef"]))
    return lat, lon

records = []
gps_points = []

for file in uploaded_files:
    try:
        img = Image.open(file)
        info = {
            "filename": file.name,
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "filesize_bytes": len(file.getvalue())
        }
        tags = exifread.process_file(file, details=False)
        for tag, value in tags.items():
            info[tag] = str(value)
        lat, lon = extract_gps(tags)
        if lat and lon:
            info["GPSLink"] = f"https://www.google.com/maps?q={lat},{lon}"
            gps_points.append((file.name, lat, lon))
        text = pytesseract.image_to_string(img).strip()
        if text:
            info["ocr_text"] = text
            if len(text)<20:
                info["ocr_language"] = "Not available"
            else:
                lang, confidence = langid.classify(text)
                info["ocr_language"] = f"{lang} ({confidence:.2f})"
        else:
            info["ocr_text"] = "Not available"
            info["ocr_language"] = "Not available"
        if "GPSLink" not in info:
            info["GPSLink"] = "Not available"
        records.append(info)
    except Exception as e:
        records.append({"filename": file.name, "error": str(e), "ocr_text":"Not available","ocr_language":"Not available","GPSLink":"Not available"})

df = pd.DataFrame(records)
for col in ["GPSLink","ocr_text","ocr_language"]:
    if col not in df.columns:
        df[col] = "Not available"
    else:
        df[col] = df[col].fillna("Not available")

st.subheader("Metadata Table")
st.dataframe(df[["filename","format","width","height","filesize_bytes","ocr_text","ocr_language","GPSLink"]])

st.subheader("Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Image Format Distribution**")
    fig, ax = plt.subplots()
    df["format"].value_counts().plot(kind="bar", color="skyblue", ax=ax)
    st.pyplot(fig)

with col2:
    st.markdown("**File Size Distribution (KB)**")
    fig, ax = plt.subplots()
    (df["filesize_bytes"]/1024).plot(kind="hist", bins=10, color="salmon", ax=ax)
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Top Resolutions**")
    df["resolution"] = df["width"].astype(str)+"x"+df["height"].astype(str)
    fig, ax = plt.subplots()
    df["resolution"].value_counts().head(10).plot(kind="bar", color="lightgreen", ax=ax)
    st.pyplot(fig)

with col4:
    st.markdown("**EXIF Presence**")
    has_exif = df.filter(like="EXIF").notna().any(axis=1).sum()
    sizes = [has_exif, len(df)-has_exif]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=["With EXIF","Without EXIF"], autopct="%1.1f%%", colors=["gold","lightgrey"])
    st.pyplot(fig)

st.subheader("Download Metadata")
csv = df.to_csv(index=False).encode('utf-8')
json_data = df.to_json(orient="records", indent=2).encode('utf-8')

st.download_button("Download CSV", data=csv, file_name="metadata.csv", mime="text/csv")
st.download_button("Download JSON", data=json_data, file_name="metadata.json", mime="application/json")

if gps_points:
    st.subheader("Detected GPS Coordinates")
    for f, lat, lon in gps_points:
        st.markdown(f"- {f}: [{lat}, {lon}](https://www.google.com/maps?q={lat},{lon})")
