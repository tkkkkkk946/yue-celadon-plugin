from torchvision import models, transforms
from PIL import Image
import torch
from torch import nn
from torchvision.models import ResNet50_Weights
import os

# Define the path to best_model.pth
MODEL_PATH = "best_model.pth"

# Verify the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure train_model.py has been run to generate best_model.pth.")

# Load model with the same architecture as in train_model.py
def create_model(num_classes):
    model = models.resnet50(weights=None)  # No pretrained weights since we load custom model
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

# Initialize model
num_classes = 8  # Match the number of classes from train_model.py
model = create_model(num_classes)

# Load the state_dict with weights_only=True
checkpoint = torch.load(MODEL_PATH, weights_only=True)
model.load_state_dict(checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Transform (same as in original main.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names (hardcoded to match train_model.py's train_dataset.classes)
class_names = ['2.碗、盘、碟', '3.壶', '4.瓶', '5.罐、翁', '6.尊、炉', '7.香薰、灯盏', '8.文房用具', 'cup']

def predict(image_path):
    img = Image.open(image_path).convert('RGB')  # Ensure RGB format
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
    return predicted_class, class_names[predicted_class]