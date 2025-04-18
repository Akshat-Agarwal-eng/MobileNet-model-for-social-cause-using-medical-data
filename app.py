import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import io

# 🔧 Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 📂 Paths
model_path = "best_mobilenet_v3_small_dysgraphia.pth"
base_dir = "C:/Users/aksha/OneDrive/Desktop/DATASET DYSGRAPHIA HANDWRITING"  # Update if server path differs

# 📸 Transformations
IMG_SIZE = 224
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 🧠 Load model
def load_model():
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-2:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1)
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 🔍 Feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

# 📊 Precompute dataset features
def compute_dataset_features(model, dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} not found")
    dataset = datasets.ImageFolder(dataset_path, transform=val_test_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    class_names = dataset.classes
    feature_dict = {class_names[0]: [], class_names[1]: []}
    
    extractor = FeatureExtractor(model).to(device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = extractor(images)
            for feat, label in zip(features, labels):
                feature_dict[class_names[label.item()]].append(feat.cpu().numpy())
    
    for cls in feature_dict:
        feature_dict[cls] = np.array(feature_dict[cls])
    return feature_dict, class_names

# 🔮 Predict dysgraphia
def predict_dysgraphia(model, image_bytes, feature_dict, class_names):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")
    
    image_tensor = val_test_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()  # Already a Python float
    
    prediction = 'Potential Dysgraphia' if prob > 0.5 else 'Low Potential Dysgraphia'
    
    extractor = FeatureExtractor(model).to(device)
    with torch.no_grad():
        input_features = extractor(image_tensor).cpu().numpy()
    
    similarities = {}
    for cls in class_names:
        if len(feature_dict[cls]) > 0:
            sim = cosine_similarity(input_features, feature_dict[cls]).mean()
            similarities[cls] = float(sim)  # Convert float32 to Python float
        else:
            similarities[cls] = 0.0
    
    result = {
        "prediction": prediction,
        "confidence": round(prob * 100, 2),  # Python float
        "similarity": {
            class_names[0]: similarities[class_names[0]],  # Already converted
            class_names[1]: similarities[class_names[1]]
        },
        "warning": ""
    }
    
    if prediction == 'Potential Dysgraphia' and prob > 0.8:
        if similarities[class_names[0]] > similarities[class_names[1]]:
            result["warning"] = "High confidence for Potential Dysgraphia, but similarity suggests Low Potential Dysgraphia."
    elif prediction == 'Low Potential Dysgraphia' and prob < 0.2:
        if similarities[class_names[1]] > similarities[class_names[0]]:
            result["warning"] = "High confidence for Low Potential Dysgraphia, but similarity suggests Potential Dysgraphia."
    if 0.4 < prob < 0.6:
        result["warning"] = "Low confidence prediction. Result may be unreliable."
    if max(similarities.values()) < 0.5:
        result["warning"] += " Low similarity to dataset. Result may be less reliable."

    return result

# Flask app
app = Flask(__name__)

# Load model and features at startup
try:
    model = load_model()
    feature_dict, class_names = compute_dataset_features(model, base_dir)
except Exception as e:
    print(f"Startup error: {str(e)}")
    exit(1)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Handwriting Dysgraphia Checker</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <input type="submit" value="Upload and Predict">
        </form>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        image_bytes = image_file.read()
        result = predict_dysgraphia(model, image_bytes, feature_dict, class_names)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)