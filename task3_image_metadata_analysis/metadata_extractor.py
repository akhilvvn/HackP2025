import os
import json
import pandas as pd
from PIL import Image
import exifread
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import langid

input_dir = "samples/images"
output_dir = "outputs"
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def extract_gps(tags):
    def conv(values, ref):
        d, m, s = [float(v.num) / float(v.den) for v in values]
        coord = d + m/60.0 + s/3600.0
        if ref in ['S', 'W']:
            coord *= -1
        return coord
    lat = lon = None
    if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
        lat = conv(tags["GPS GPSLatitude"].values, str(tags["GPS GPSLatitudeRef"]))
        lon = conv(tags["GPS GPSLongitude"].values, str(tags["GPS GPSLongitudeRef"]))
    return lat, lon

records = []
gps_points = []

for file in os.listdir(input_dir):
    path = os.path.join(input_dir, file)
    if not os.path.isfile(path):
        continue
    try:
        img = Image.open(path)
        info = {
            "filename": file,
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "filesize_bytes": os.path.getsize(path)
        }
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
            for tag, value in tags.items():
                info[tag] = str(value)
            lat, lon = extract_gps(tags)
            if lat and lon:
                info["GPSLatitude"] = lat
                info["GPSLongitude"] = lon
                info["GPSLink"] = f"https://www.google.com/maps?q={lat},{lon}"
                gps_points.append((file, lat, lon))
        try:
            text = pytesseract.image_to_string(img).strip()
            if text:
                info["ocr_text"] = text
                if len(text) < 20:
                    info["ocr_language"] = "uncertain"
                else:
                    lang, confidence = langid.classify(text)
                    info["ocr_language"] = f"{lang} ({confidence:.2f})"
            else:
                info["ocr_text"] = ""
                info["ocr_language"] = ""
        except Exception as e:
            info["ocr_error"] = str(e)
        records.append(info)
    except Exception as e:
        records.append({"filename": file, "error": str(e)})

df = pd.DataFrame(records)
csv_path = os.path.join(output_dir, "metadata.csv")
json_path = os.path.join(output_dir, "metadata.json")
df.to_csv(csv_path, index=False)
df.to_json(json_path, orient="records", indent=2)

format_counts = df["format"].value_counts()
plt.figure(figsize=(6,4))
format_counts.plot(kind="bar", color="skyblue")
plt.title("Image Format Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "format_distribution.png"))
plt.close()

plt.figure(figsize=(6,4))
df["filesize_bytes"].dropna().apply(lambda x: x/1024).plot(kind="hist", bins=10, color="salmon")
plt.title("File Size Distribution (KB)")
plt.xlabel("KB")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "filesize_distribution.png"))
plt.close()

plt.figure(figsize=(6,4))
df["resolution"] = df["width"].astype(str) + "x" + df["height"].astype(str)
df["resolution"].value_counts().head(10).plot(kind="bar", color="lightgreen")
plt.title("Top Resolutions")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "resolution_distribution.png"))
plt.close()

has_exif = df.filter(like="EXIF").notna().any(axis=1).sum()
labels = ["With EXIF", "Without EXIF"]
sizes = [has_exif, len(df) - has_exif]
plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["gold","lightgrey"])
plt.title("EXIF Presence")
plt.savefig(os.path.join(plots_dir, "exif_presence.png"))
plt.close()

report_path = os.path.join(output_dir, "report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("<html><head><title>Image Metadata Report</title></head><body>")
    f.write("<h1>Image Metadata Report</h1>")
    f.write("<h2>Overview</h2>")
    f.write(f"<p>Total images: {len(df)}</p>")
    f.write("<h2>Plots</h2>")
    for plot in ["format_distribution.png", "filesize_distribution.png", 
                 "resolution_distribution.png", "exif_presence.png"]:
        f.write(f"<h3>{plot.replace('_',' ').title()}</h3>")
        f.write(f'<img src="plots/{plot}" style="max-width:600px;"><br>')
    f.write("<h2>Metadata Table</h2>")

    f.write("<table border='1' cellspacing='0' cellpadding='4'>")
    f.write("<tr><th>Filename</th><th>Format</th><th>Resolution</th><th>Filesize (KB)</th>"
            "<th>OCR Text</th><th>Detected Language</th><th>GPS</th></tr>")
    for _, row in df.iterrows():
        gps_html = ""
        if "GPSLink" in row and row["GPSLink"]:
            gps_html = f'<a href="{row["GPSLink"]}" target="_blank">View on Maps</a>'
        f.write("<tr>")
        f.write(f"<td>{row['filename']}</td>")
        f.write(f"<td>{row.get('format','')}</td>")
        f.write(f"<td>{row.get('width','')}x{row.get('height','')}</td>")
        f.write(f"<td>{round(row.get('filesize_bytes',0)/1024,2)}</td>")
        f.write(f"<td>{row.get('ocr_text','').replace('<','&lt;').replace('>','&gt;')[:200]}</td>")
        f.write(f"<td>{row.get('ocr_language','')}</td>")
        f.write(f"<td>{gps_html}</td>")
        f.write("</tr>")
    f.write("</table>")

    f.write("</body></html>")


print(f"Metadata saved to {csv_path} and {json_path}")
print(f"Plots saved to {plots_dir}")
print(f"HTML report generated at {report_path}")
if gps_points:
    print("GPS coordinates extracted:")
    for f, lat, lon in gps_points:
        print(f"  {f}: {lat}, {lon} -> https://www.google.com/maps?q={lat},{lon}")
