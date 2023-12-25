import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from model import VGG16
from dataset import CustomImageDataset
from torch.utils.data import DataLoader


# 定义数据集目录
train_dir = 'train/'
validation_dir = 'validation/'
NUM_CLASSES = 1000
train_label_csv = ''
val_label_csv = ''

# 数据预处理,也可自己定义
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 读取数据集
train_data = CustomImageDataset(train_label_csv, train_dir, transform=transform)
validation_data = CustomImageDataset(val_label_csv, validation_dir, transform=transform)

# 数据加载器

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)

# 构建模型
model = VGG16(num_classes=NUM_CLASSES)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义指标记录
best_loss = float('inf')
best_acc = 0.0

# 模型训练和验证
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, train loss: {running_loss/len(train_dataloader)}')

    # 在验证集上评估模型
    correct = 0
    total = 0
    validation_loss = 0.0
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels.float().view(-1, 1))
            validation_loss += loss.item()
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels.float().view(-1, 1)).sum().item()

    # 计算指标并保存最优模型
    acc = 100 * correct / total
    avg_train_loss = running_loss/len(train_dataloader)
    avg_validation_loss = validation_loss/len(val_dataloader)

    if avg_validation_loss < best_loss:
        best_loss = avg_validation_loss
        torch.save(model.state_dict(), 'model_best_loss.pth')
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'model_best_acc.pth')
    print(f'Epoch {epoch + 1}, validation loss: {avg_validation_loss}, accuracy: {acc}%')

# 保存最终模型
torch.save(model.state_dict(), 'model_final.pth')