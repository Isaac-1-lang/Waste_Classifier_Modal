import os
import gc
import json
import requests
import sys
from datetime import datetime
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification

print("Starting application...")
print(f"Python version: {sys.version}")
print(f"PORT environment variable: {os.environ.get('PORT', 'Not set')}")

# Create Flask app first
app = Flask(__name__)
CORS(app)

# Simple health check that works immediately
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Waste Classification API is live!",
        "port": os.environ.get('PORT', 'Not set'),
        "python_version": sys.version
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Server is running",
        "port": os.environ.get('PORT', 'Not set')
    })

# Initialize ML components after Flask app is created
print("Loading ML model...")
try:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processor and model - with error handling
    processor = AutoImageProcessor.from_pretrained("Claudineuwa/waste_classifier_Isaac")
    model = AutoModelForImageClassification.from_pretrained("Claudineuwa/waste_classifier_Isaac").to(device)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Label mapping
    id2label = model.config.id2label
    
    print("Model loaded successfully!")
    MODEL_LOADED = True
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    MODEL_LOADED = False
    device = "cpu"
    model = None
    processor = None
    id2label = {}

# Backend URL configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "https://trash2treasure-backend.onrender.com/wasteSubmission")

# Update health check to include model status
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "device": str(device),
        "backend_url": BACKEND_URL,
        "port": os.environ.get('PORT', 'Not set'),
        "message": "Server is running successfully"
    })

# PREDICTION ROUTE - Only if model is loaded
@app.route("/predict", methods=["POST"])
def predict_image():
    if not MODEL_LOADED:
        return jsonify({
            "error": "Model not loaded",
            "success": False
        }), 503
    
    print("Received request to /predict")
    
    if "image" not in request.files:
        print("No image in request")
        return jsonify({"error": "No image uploaded", "success": False}), 400

    file = request.files["image"]
    print(f"Received image: {file.filename}")

    try:
        image = Image.open(file).convert("RGB")
        print(f"Image loaded: {image.size}")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
        
        result = id2label[pred.item()]
        confidence = conf.item()
        
        print(f"Prediction: {result}, Confidence: {confidence:.4f}")
        
        # Prepare data to send to backend
        classification_data = {
            "prediction": result,
            "confidence": f"{confidence:.4f}",
            "timestamp": datetime.now().isoformat(),
            "image_filename": file.filename,
            "model_version": "Claudineuwa/waste_classifier_Isaac",
            "device": str(device)
        }
        
        # Send data to backend
        try:
            backend_response = requests.post(
                f"{BACKEND_URL}/waste-classification",
                json=classification_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if backend_response.status_code == 200:
                print("Data successfully sent to backend")
                backend_result = backend_response.json()
            else:
                print(f"Backend returned status {backend_response.status_code}")
                backend_result = {"backend_status": "error", "message": "Failed to send to backend"}
                
        except requests.exceptions.RequestException as e:
            print(f"Error sending to backend: {str(e)}")
            backend_result = {"backend_status": "error", "message": f"Backend connection failed: {str(e)}"}
        
        return jsonify({
            "prediction": result,
            "confidence": f"{confidence:.4f}",
            "success": True,
            "message": "Classification successful",
            "backend_response": backend_result
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            "error": f"Classification failed: {str(e)}", 
            "success": False
        }), 500

    finally:
        # Clean up memory
        if MODEL_LOADED:
            for var in ['inputs', 'outputs', 'probs', 'conf', 'pred', 'image']:
                if var in locals():
                    del locals()[var]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ERROR HANDLERS
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Route not found",
        "available_routes": [
            "GET  /",
            "GET  /health", 
            "POST /predict"
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "Check the HTTP method (GET/POST) for this route"
    }), 405

if __name__ == "__main__":
    # Get port from environment - this is CRITICAL for Render
    port = int(os.environ.get("PORT", 10000))
    
    print("=" * 50)
    print(" Starting Waste Classification API")
    print("=" * 50)
    print(f"Server starting on 0.0.0.0:{port}")
    print(f"Model loaded: {MODEL_LOADED}")
    print(f"Device: {device}")
    print(f"Backend URL: {BACKEND_URL}")
    print("=" * 50)
    
    # CRITICAL: Use threaded=True for better performance on Render
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=False,
        threaded=True,
        use_reloader=False  # Prevent multiple restarts
    )
