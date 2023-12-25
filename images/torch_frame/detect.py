import torch
import torchvision.transforms as transforms
from PIL import Image
from model import VGG16  

best_pt_path = ''
detect_image_file_path = ''

# 加载预训练的 VGG16 模型
model = VGG16(num_classes=1000)  # 假设预训练模型有 1000 个类别
model.load_state_dict(torch.load(best_pt_path))  # 加载预训练权重
model.eval()  # 设置模型为评估模式

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取图像并进行预处理
image = Image.open(detect_image_file_path)  # 替换为实际的图像文件名
image = transform(image)
image = image.unsqueeze(0)  # 添加 batch 维度

# 使用模型进行预测
with torch.no_grad():
    output = model(image)

# 获取预测结果
_, predicted = torch.max(output, 1)
print('Predicted class:', predicted.item())