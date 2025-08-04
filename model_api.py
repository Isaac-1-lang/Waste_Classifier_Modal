import os
import gc
import json
import requests
from datetime import datetime
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = AutoImageProcessor.from_pretrained("Claudineuwa/waste_classifier_Isaac")
model = AutoModelForImageClassification.from_pretrained("Claudineuwa/waste_classifier_Isaac").to(device)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Label mapping
id2label = model.config.id2label

# Backend URL configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "https://trash2treasure-backend.onrender.com/wasteSubmission")

app = Flask(__name__)

# ROOT ROUTE - Test if server is working
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Waste Classification API is running successfully!",
        "server_info": {
            "device": str(device),
            "model_loaded": model is not None,
            "backend_url": BACKEND_URL,
            "routes_available": [
                "GET  / - This route (server status)",
                "GET  /health - Health check",
                "POST /predict - Image classification and send to backend"
            ]
        }
    })

# HEALTH CHECK ROUTE
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None,
        "backend_url": BACKEND_URL,
        "message": "Server is running successfully",
        "version": "2.0"
    })

# PREDICTION ROUTE - Classify image and send to backend
@app.route("/predict", methods=["POST"])
def predict_image():
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

# Handle CORS for React Native (if needed)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    print("=" * 50)
    print(" Starting Waste Classification API")
    print("=" * 50)
    print(f"Server URL: http://0.0.0.0:{port}")
    print(f"External URL: http://192.168.0.109:{port}")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Device: {device}")
    print(f"Model loaded: {model is not None}")
    print("\nAvailable Routes:")
    print(f"• GET  http://192.168.0.109:{port}/")
    print(f"• GET  http://192.168.0.109:{port}/health")
    print(f"• POST http://192.168.0.109:{port}/predict")
    print("\nReady to classify images and send data to backend!")
    print("=" * 50)
    
    app.run(host="0.0.0.0", port=port, debug=True)
