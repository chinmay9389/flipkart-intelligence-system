import joblib
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# 1. Load Return Risk Model Tool
def check_return_risk(order_features: dict) -> dict:
    artifact = joblib.load("models/return_risk_model.pkl")
    model = artifact["model"]
    t_star = artifact["t_star_rf"]
    
    df_input = pd.DataFrame([order_features])
    prob = float(model.predict_proba(df_input)[0, 1])
    
    # Dynamic bucket cutpoints anchored to t*_rf
    if prob < t_star:
        risk_bucket = "Low"
    elif prob >= t_star + 0.15:
        risk_bucket = "High"
    else:
        risk_bucket = "Medium"
        
    return {
        "return_probability": round(prob, 4),
        "risk_bucket": risk_bucket,
        "anchored_threshold_t_star": t_star
    }

# 2. Classify Product Image Tool
def classify_product_image(image_path: str) -> dict:
    class_names = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot"]
    
    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("models/product_classifier.pt", map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_class = torch.max(probs, dim=0)
        
    return {
        "predicted_category": class_names[pred_class.item()],
        "confidence": round(float(conf), 4),
        "image_path": image_path
    }
