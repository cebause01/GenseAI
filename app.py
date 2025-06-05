from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import time
from ultralytics import YOLO

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model using Ultralytics API
def load_model():
    try:
        model_path = 'C://Workspace//GenseAI//Basic-1st.pt'
        print("Attempting to load YOLO model from:", model_path)
        if not os.path.exists(model_path):
            print(f"Model file does not exist at: {model_path}")
            return None
        model = YOLO(model_path)
        print("YOLO model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

model = load_model()

# Process prediction using Ultralytics YOLO API
def process_prediction(image_path):
    try:
        if model is None:
            print("Error: Model is not loaded.")
            return [{'message': 'Model is not loaded. Check server logs for details.'}]
        print("Running YOLO inference on uploaded image.")
        results = model(image_path)  # list of Results objects
        print(f"Raw YOLO results: {results}")
        detections = []
        for result in results:
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'xyxy') and len(result.boxes.xyxy) > 0:
                print(f"Boxes found: {len(result.boxes.xyxy)}")
                for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    class_name = result.names[int(cls)] if hasattr(result, 'names') else str(int(cls))
                    detections.append({
                        'class': class_name,
                        'confidence': float(score),
                        'box': box.cpu().numpy().tolist()
                    })
            else:
                print("No boxes detected in result.")
        print(f"Detections: {detections}")
        return detections if detections else [{'message': 'No objects detected'}]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return [{'message': f'Error during prediction: {e}'}]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        detections = process_prediction(filepath)
        
        return jsonify({
            'filename': filename,
            'detections': detections
        })
    
    return jsonify({'error': 'File type not allowed'})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)