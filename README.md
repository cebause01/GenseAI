# GenseAI

Flask web app for image upload and YOLO-based object detection.

## Stack

- Python, Flask
- Ultralytics YOLO
- Pillow

## Setup

```bash
pip install flask ultralytics pillow
python app.py
```

Place the YOLO weights file (`Basic-1st.pt`) in the project root and update the model path in `app.py` if needed.

## Features

- Image upload endpoint
- YOLO inference on uploaded images
- Web UI via `index.html`
