import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# 用于显示进度条
from tqdm import tqdm
import seaborn as sns
import requests
import zipfile
import os

# 定义URL和文件名（下面的地址改为实际训练数据的地址）
url = "http://xxx/classify-leaves.zip"
filename = "classify-leaves.zip"

# 下载文件
response = requests.get(url)

# 将下载的内容保存到文件中
with open(filename, "wb") as f:
    f.write(response.content)

# 解压缩文件内容
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall()

# 移除已下载的zip文件
os.remove(filename)

# 看看label文件长啥样
labels_dataframe = pd.read_csv('./train.csv')

# 把label文件排个序
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
print(n_classes)
print(leaves_labels[:10])

# 把label转成对应的数字
class_to_num = dict(zip(leaves_labels, range(n_classes)))

# 再转换回来，方便最后预测的时候使用
num_to_class = {v : k for k, v in class_to_num.items()}

# 继承pytorch的dataset，创建自己的
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """

        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # 读取 csv 文件
        self.data_info = pd.read_csv(csv_path, header=None)
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]

        img_as_img = Image.open(self.file_path + single_image_name)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            label = self.label_arr[index]
            number_label = class_to_num[label]

            return img_as_img, number_label

    def __len__(self):
        return self.real_len

train_path = './train.csv'
test_path = './test.csv'
img_path = './'

train_dataset = LeavesData(train_path, img_path, mode='train')
val_dataset = LeavesData(train_path, img_path, mode='valid')
test_dataset = LeavesData(test_path, img_path, mode='test')

# 定义data loader
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=5
    )

val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=5
    )
test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=5
    )

# 确定是在CPU还是GPU上
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
print(device)

# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False

# resnet34模型
def res_model(num_classes, feature_extract = False, use_pretrained=True):

    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes)
    )
    return model_ft

# resnext50模型
def resnext_model(num_classes, feature_extract = False, use_pretrained=True):

    model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

# 超参数
learning_rate = 3e-4
weight_decay = 1e-3
num_epoch = 50
model_path = './pre_resnext_model.ckpt'

# 初始化模型并将其放在指定的设备上
model = resnext_model(176)
model = model.to(device)
model.device = device
# 对于分类任务，我们使用交叉熵作为性能度量
criterion = nn.CrossEntropyLoss()

# 初始化优化器，您可以调整一些超参数，如学习率
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)

# 训练轮数
n_epochs = num_epoch

best_acc = 0.0
for epoch in range(n_epochs):
    # ---------- 训练 ----------
    # 确保模型在训练之前处于训练模式
    model.train()
    # 记录训练过程中的信息
    train_loss = []
    train_accs = []
    i = 0
    # 遍历训练集的批次
    for batch in tqdm(train_loader):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        # 前向传播数据
        logits = model(imgs)
        # 计算交叉熵损失
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        # 计算参数的梯度
        loss.backward()
        # 使用计算出的梯度来更新参数
        optimizer.step()
        # 更新学习率
        scheduler.step()
        if(i % 500 == 0):
            print("learning_rate:", scheduler.get_last_lr()[0])
        i = i + 1

        # 计算当前批次的准确率
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        # 记录损失和准确率
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # 打印信息
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- 验证 ----------
    # 确保模型在评估模式下，这样一些模块如dropout就会被禁用并正常工作
    model.eval()
    # 记录验证过程中的信息
    valid_loss = []
    valid_accs = []

    # 遍历验证集的批次
    for batch in tqdm(val_loader):
        imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))

        loss = criterion(logits, labels.to(device))

        # 计算当前批次的准确率
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # 记录损失和准确率
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # 打印信息
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # 如果模型性能提高，保存此时的模型
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))