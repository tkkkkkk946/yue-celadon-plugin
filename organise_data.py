import os
import shutil

data_dir = 'data'  # Data root directory
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
output_train_dir = os.path.join(data_dir, 'train_reorganized')
output_val_dir = os.path.join(data_dir, 'val_reorganized')

os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)

def reorganize_data(source_dir, target_dir):
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            for sub_class_name in os.listdir(class_path):
                sub_class_path = os.path.join(class_path, sub_class_name)
                if os.path.isdir(sub_class_path):
                    new_class_name = f"{class_name}_{sub_class_name}"  # Combine names
                    new_class_dir = os.path.join(target_dir, new_class_name)
                    os.makedirs(new_class_dir, exist_ok=True)
                    for img_file in os.listdir(sub_class_path):
                        img_file_path = os.path.join(sub_class_path, img_file)
                        if img_file.endswith(('.jpg', '.jpeg', '.png')):
                            shutil.move(img_file_path, os.path.join(new_class_dir, img_file))

reorganize_data(train_dir, output_train_dir)
reorganize_data(val_dir, output_val_dir)
print("Data organization completed!")
