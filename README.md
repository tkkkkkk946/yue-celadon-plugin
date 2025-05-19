越青瓷分类项目

本项目是一个基于 PyTorch 的图像分类项目，主要用于对越青瓷的不同类型进行分类。项目包含数据整理、模型训练、模型推理以及提供 API 服务等功能。

项目结构

organise_data.py：用于对数据集进行重新组织，将原始数据集按照特定规则进行分类整理。
train_model.py：使用 ResNet50 模型对整理后的数据集进行训练，并保存最佳模型。
main.py：加载训练好的模型，对单张图像进行分类预测。
api.py：使用 FastAPI 搭建一个简单的 API 服务，接收图像文件并返回分类结果。

环境要求

Python 3.x

PyTorch

torchvision

Pillow

tqdm

fastapi

uvicorn

你可以使用以下命令安装所需的依赖：

pip install torch torchvision pillow tqdm fastapi uvicorn

使用步骤
1. 数据整理
运行 organise_data.py 脚本，将数据集按照特定规则进行重新组织：
python train_model.py
2. 模型训练
运行 train_model.py 脚本，使用整理后的数据集对模型进行训练：
3. 单张图像预测
运行 main.py 脚本，对单张图像进行分类预测：
python main.py
4. 启动 API 服务
运行以下命令启动 FastAPI 服务：
uvicorn api:app --reload
