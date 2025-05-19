import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torchvision.models import ResNet50_Weights

# 增强数据变换，添加更多随机性
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

from PIL import Image

def check_and_fix_image(image_path):
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error with {image_path}: {e}")
        return None

# 数据路径
data_dir = r'C:\Users\30385\Desktop\yue_celadon_plugin'
train_dir = os.path.join(data_dir, 'data', 'train')
val_dir = os.path.join(data_dir, 'data', 'val')

# Check if data directories exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found at {train_dir}. Please ensure the dataset is correctly set up.")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory not found at {val_dir}. Please ensure the dataset is correctly set up.")

# 创建带Dropout的模型
def create_model(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # Use weights instead of pretrained
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

def main():
    # Load datasets
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
        val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    except Exception as e:
        raise RuntimeError(f"Failed to load datasets: {str(e)}. Ensure the train and val directories contain subdirectories with images.")

    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty at {train_dir}. Ensure there are images in the class subdirectories.")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty at {val_dir}. Ensure there are images in the class subdirectories.")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"训练类别: {class_names}")

    model = create_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
        best_acc = 0.0
        train_losses, train_accs, val_accs = [], [], []
        # Define absolute path for saving the model
        model_path = r'C:\Users\30385\Desktop\yue_celadon_plugin\best_model.pth'
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            print(f"Epoch {epoch+1}/{num_epochs}, 训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.4f}")

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    correct += torch.sum(preds == labels).item()
                    total += labels.size(0)
            
            val_acc = correct / total
            val_accs.append(val_acc)
            print(f"验证准确率: {val_acc:.4f}")
            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print(f"已保存最佳模型 at {model_path}, 准确率: {best_acc:.4f}")
        
        return model, best_acc, {'train_loss': train_losses, 'train_acc': train_accs, 'val_acc': val_accs}

    print(f"开始训练，训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    try:
        trained_model, best_accuracy, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)
        print(f"\n训练完成！最佳验证准确率: {best_accuracy:.4f}")
        model_path = r'C:\Users\30385\Desktop\yue_celadon_plugin\best_model.pth'
        print(f"模型已保存至: {model_path}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()