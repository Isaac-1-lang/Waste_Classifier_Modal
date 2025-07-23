from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "waste_classifier")


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

# Check if the model path exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

# Load model and processor with local_files_only=True
try:
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_PATH, 
        local_files_only=True
    )
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_PATH, 
        local_files_only=True
    )
    model.eval()
    print("Model and processor loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def predict_image(image_bytes, model, image_processor, device="cpu"):
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

@app.route('/classify', methods=['POST'])
def classify():
    results = []
    files = request.files.getlist('images')
    for file in files:
        image_bytes = file.read()
        result = predict_image(image_bytes, model, image_processor)
        results.append(result)
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)