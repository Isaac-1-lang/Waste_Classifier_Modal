from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Fix the model path - should match your submodule name
MODEL_PATH = os.path.join(os.path.dirname(__file__), "waste_classifier_Isaac")

LABEL2INFO = {
    0: {
        "label": "biodegradable",
        "description": "Easily breaks down naturally. Good for composting.",
        "recyclable": False,
        "disposal": "Use compost or organic bin",
        "example_items": ["banana peel", "food waste", "paper"],
        "environmental_benefit": "Composting biodegradable waste returns nutrients to the soil, reduces landfill use, and lowers greenhouse gas emissions.",
        "protection_tip": "Compost at home or use municipal organic waste bins. Avoid mixing with plastics or hazardous waste.",
        "poor_disposal_effects": "If disposed of improperly, biodegradable waste can cause methane emissions in landfills and contribute to water pollution and eutrophication."
    },
    1: {
        "label": "non_biodegradable",
        "description": "Does not break down easily. Should be disposed of carefully.",
        "recyclable": False,
        "disposal": "Use general waste bin or recycling if possible",
        "example_items": ["plastic bag", "styrofoam", "metal can"],
        "environmental_benefit": "Proper disposal and recycling of non-biodegradable waste reduces pollution, conserves resources, and protects wildlife.",
        "protection_tip": "Reduce use, reuse items, and recycle whenever possible. Never burn or dump in nature.",
        "poor_disposal_effects": "Improper disposal leads to soil and water pollution, harms wildlife, and causes long-term environmental damage. Plastics can persist for hundreds of years."
    }
}

# Global variables for model and processor
model = None
image_processor = None

def load_model():
    """Load the model with proper error handling"""
    global model, image_processor
    
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    
    # Check if the model path exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model path does not exist: {MODEL_PATH}")
        # List available directories for debugging
        current_dir = os.path.dirname(__file__)
        available_dirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
        logger.info(f"Available directories: {available_dirs}")
        raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

    # Load model and processor with local_files_only=True
    try:
        logger.info("Loading model...")
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_PATH, 
            local_files_only=True
        )
        logger.info("Loading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(
            MODEL_PATH, 
            local_files_only=True
        )
        model.eval()
        logger.info("Model and processor loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def predict_image(image_bytes, device="cpu"):
    """Predict image classification"""
    if model is None or image_processor is None:
        raise RuntimeError("Model not loaded properly")
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            label_id = pred.item()
            confidence = conf.item()
        
        info = LABEL2INFO[label_id].copy()
        info["confidence"] = round(confidence, 2)
        info["eco_points_earned"] = 10  # Dummy value
        return info
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/classify', methods=['POST'])
def classify():
    """Classification endpoint"""
    if model is None or image_processor is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        results = []
        files = request.files.getlist('images')
        
        if not files:
            return jsonify({"error": "No images provided"}), 400
        
        for file in files:
            if file.filename == '':
                continue
            image_bytes = file.read()
            result = predict_image(image_bytes)
            results.append(result)
        
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error in classify: {e}")
        return jsonify({"error": str(e)}), 500

# Initialize the model when the app starts
logger.info("Starting Flask app...")
model_loaded = load_model()

if not model_loaded:
    logger.warning("App starting without model - some features may not work")

if __name__ == '__main__':
    # Use environment PORT for deployment, fallback to 5000 for local
    port = int(os.environ.get("PORT", 5000))
    # Bind to 0.0.0.0 for deployment, disable debug in production
    app.run(host="0.0.0.0", port=port, debug=False)