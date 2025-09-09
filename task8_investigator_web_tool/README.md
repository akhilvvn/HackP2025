# Investigator Web Tool

## Task Description

This project implements a **web-based investigative image editor** with a **FastAPI backend** and a **React + Vite + Konva frontend**. It is designed for **forensics, journalism, and cybersecurity use cases** where sensitive image content must be **anonymized, cropped, or highlighted** before sharing or archiving. The tool provides an **interactive canvas editor** powered by **React + Konva**, enabling users to blackout, blur, crop or emphasize specific regions of an image. Combined with **YOLOv8 object detection**, it can assist investigators in automatically identifying objects of interest for faster editing. Features implemented include:

- **Image Upload & Preview** – secure file handling with FastAPI.  
- **Editing Tools** – mark and anonymize regions with precision.  
- **Tool Modes**:  
- **Select Tool** – draw zones, select, resize, and move them  
- **Deselect Tool** – draw zones to clear/remove selections  
- **Crop Tool** – crop images to the desired region  
- **Hand Tool** – pan and zoom the canvas
- **Blackout** – Hide sensitive areas.  
- **Blur** – Anonymize faces/objects.  
- **Preview in Real-Time** – See edits before exporting.  
- **Zoom & Pan Support** – precise inspection of image details. 
- **AI-assisted object detection (YOLOv8)** – optional automatic object selection.  
- **Export & Download** – Processed images saved back to the user.

The goal is to provide a **lightweight, browser-accessible tool** for investigative workflows, combining **manual editing precision** with **AI-assisted suggestions**.

---

## Project Structure

HackP2025/
│── task8_investigator_web_tool/
│   ├── README.md
│   ├── backend/
│   │   ├── requirements.txt           # Python dependencies
│   │   ├── main.py                    # FastAPI backend (API + AI image processing)
│   │   ├── yolov8l.pt                 # YOLOv8 model weight
│   │
│   ├── frontend/
│   │   ├── package.json               # Frontend dependencies & npm/yarn scripts
│   │   ├── vite.config.js             # Vite bundler configuration
│   │   ├── public/                    # Public assets
│   │   ├── node_modules/              # Installed npm packages
│   │   └── src/
│   │       ├── App.jsx                # Main React component
│   │       ├── App.css                # Global styles for App.jsx
│   │       ├── CanvasEditor.jsx       # Core image editing UI
│   │       ├── CanvasEditor.css       # Styling for CanvasEditor
│   │       ├── api.js                 # Handles API requests to FastAPI backend
│   │       ├── main.jsx               # React entry point
│   │       ├── index.css              # Global styling, resets & base CSS
│   │       └── assets/
│   │
│   └── samples/
│       └── sample_images/
 

---

## Installation

## Backend (FastAPI)

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

### Install dependencies

```
cd backend

pip install -r requirements.txt

```

**Note:** Ensure YOLOv8 model weights are available. This project uses yolov8l.pt inside backend/. If missing, Ultralytics will automatically download it the first time you run the app. Alternatively, you can manually download it from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt


## Frontend (React + Vite + Konva)

Ensure **Node.js v18+** (or newer) is installed. 

### Install dependencies

From inside the project folder, run: 

```
cd frontend

npm install

```

---

## Usage

### Start the backend. Run the FastAPI server:

```
uvicorn main:app --reload

```

By default, the backend will run on http://127.0.0.1:8000.

### Start the frontend (React + Vite app)

```
npm run dev

```

By default, the frontend will run on http://localhost:5173, open the link in your browser.

- Upload an image for editing

- The image will automatically fit into the editor canvas.

- Navigate the canvas: Use the Hand Tool to pan and zoom for detailed inspection. Mouse wheel or trackpad can zoom in/out.

- Mark regions: Select Tool → Use it to select the areas you want to anonymize or crop. Deselect Tool → use it to mark areas that should remain visible (undo part of a selection).

- Selected regions can be moved, resized, or removed.

- Crop the image: Switch to the Crop Tool to cut out a specific region. Cropped output replaces the canvas preview.

- Apply anonymization: Choose Blackout (solid fill) or Blur (Gaussian blur). The selected regions will be processed securely via the backend.

- Use the **Undo button** to revert your last action.

- AI-assisted object detection (optional): Click Detect Objects to automatically highlight objects in the image using YOLOv8.

- Refine or adjust detections manually if needed.

- Download the result using the download button

---

## Observations

- **Frontend–backend split**: Editing actions (draw, move, crop) happen instantly in the browser, while heavier tasks (blur, blackout, AI detection) are handled server-side for performance.  
- **Tool-based editing**: Dedicated **Select, Deselect, and Crop tools** give precise control, minimizing accidental changes.  
- **Undo support**: Users can revert mistakes instantly, ensuring a smooth investigative workflow.  
- **Auto-fit on upload**: Every image is scaled to fit the canvas, avoiding misalignment or clipping.  
- **Zoom & pan workflow**: Investigators can inspect tiny details without losing overall context.  
- **AI-assisted object detection**: YOLOv8 automatically highlights potential areas of interest, speeding up editing while leaving room for manual correction.  
- **Dark investigative theme**: Neon-dark palette creates a “forensic lab desk” feel, reinforcing the tool’s investigative purpose.  
- **Responsive design**: Layout adapts gracefully to different screen sizes, from laptops to large monitors.  
- **Download option**: Processed images can be securely exported for reporting, sharing, or archiving.  

---

## Possible Extensions

- **Advanced Anonymization**: Support for **blur and pixelation** in addition to blackout, with customizable strength levels.  
- **Flexible Selection Tools**: Add **freehand and polygon tools** for irregular or curved regions, improving precision in complex images.  
- **Batch Processing**: Upload and process **multiple images in one session**, saving time for investigative workflows.  
- **Text & OCR Integration**: Use OCR to detect and suggest sensitive text regions (IDs, license plates, documents) for anonymization.  
- **AI-Assisted Annotations**: Extend object detection beyond YOLOv8 (e.g., faces, documents, license plates) with auto-suggestions for redaction.  
- **Video Support**: Enable frame-by-frame editing or **automatic region tracking** across video sequences.  
- **Collaboration Mode**: Allow **multi-user annotations in real-time**, ideal for journalism teams or forensic analysts.  
- **Export & Reporting**: Generate **forensic-ready reports** in PDF or Markdown, including metadata (tools used, regions edited, timestamps).  
- **Customizable Themes**: Light/dark or high-contrast accessibility modes for different working environments.  

---

## Author

**Akhil V Nair** – HackP 2025