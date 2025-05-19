from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
from torchvision import models, transforms
import os
from torch import nn
from torchvision.models import ResNet50_Weights

app = FastAPI()

MODEL_PATH = r"C:\Users\30385\Desktop\yue_celadon_plugin\best_model.pth"

# 验证模型文件
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

def create_model(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=8).to(device)

# 加载模型权重
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'state_dict' in checkpoint:  # 兼容带元数据的保存格式
        model.load_state_dict(checkpoint['state_dict'])
        class_names = checkpoint.get('class_names', ['default_class']*8)
    else:
        model.load_state_dict(checkpoint)
        class_names = ['2.碗、盘、碟', '3.壶', ...]  # 默认值
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 验证输入
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Only image files are allowed")
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(400, f"Image processing failed: {str(e)}")
    finally:
        await file.close()

    # 预测
    with torch.no_grad():
        try:
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            return {
                "class_id": predicted.item(),
                "class_name": class_names[predicted.item()],
                "confidence": torch.softmax(output, dim=1)[0][predicted].item()
            }
        except Exception as e:
            raise HTTPException(500, f"Prediction error: {str(e)}")