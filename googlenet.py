import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchsummary import summary

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        super(InceptionBlock, self).__init__()

        # 1x1 卷积
        self.branch1 = nn.Conv2d(in_channels, f1, kernel_size=1)

        # 1x1 -> 3x3 卷积
        self.branch2_1 = nn.Conv2d(in_channels, f2_in, kernel_size=1)
        self.branch2_2 = nn.Conv2d(f2_in, f2_out, kernel_size=3, padding=1)

        # 1x1 -> 5x5 卷积
        self.branch3_1 = nn.Conv2d(in_channels, f3_in, kernel_size=1)
        self.branch3_2 = nn.Conv2d(f3_in, f3_out, kernel_size=5, padding=2)

        # 3x3 最大池化 -> 1x1 卷积
        self.branch4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = nn.Conv2d(in_channels, f4_out, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.branch1(x))
        b2 = F.relu(self.branch2_2(F.relu(self.branch2_1(x))))
        b3 = F.relu(self.branch3_2(F.relu(self.branch3_1(x))))
        b4 = F.relu(self.branch4_2(self.branch4_1(x)))

        return torch.cat([b1, b2, b3, b4], dim=1)  # 在通道维度拼接

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # Inception 模块
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.dropout = nn.Dropout(0.4)
        self.pool5 = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(832, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        x = self.inception5a(x)
        # x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


