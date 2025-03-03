import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PacmanNet(nn.Module):
    def __init__(self, num_actions = 5):
        super().__init__()
        # [B, 1, 42, 42]
        self.embeddings = nn.Embedding(13, 16)
        # [B, 16, 42, 42]
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # [B, 32, 42, 42]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # [B, 64, 42, 42] [B, 64, 21, 21]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # [B, 128, 21, 21] [B, 128, 11, 11]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # [B, 256, 11, 11] [B, 256, 6, 6]

        self.fc1 = nn.Linear(256 * 11 * 11, 1024)
        # self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(1024, 32)
        # self.fc2 = nn.Linear(512, 32)
        self.policy = nn.Linear(32, num_actions)
        self.value = nn.Linear(32, 1)
    
    def forward(self, x):
        B, _, H, W = x.shape
        flattened_channel = x.contiguous().reshape(B, -1).long()
        embedded_channel = self.embeddings(flattened_channel)
        embedded_channel = embedded_channel.permute(0, 2, 1).contiguous().reshape(B, 16, H, W)
        x = embedded_channel

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = x.contiguous().reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy(x), dim=1)
        value = self.value(x)
        return policy, value

class GhostNet(nn.Module):
    def __init__(self, num_actions = 125):
        super().__init__()
        # [B, 1, 42, 42]
        self.embeddings = nn.Embedding(15, 16)
        # [B, 16, 42, 42]
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # [B, 32, 42, 42]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # [B, 64, 42, 42] [B, 64, 21, 21]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # [B, 128, 21, 21] [B, 128, 11, 11]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # [B, 256, 11, 11] [B, 256, 6, 6]

        self.fc1 = nn.Linear(256 * 11 * 11, 1024)
        # self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.value1 = nn.Linear(1024, 32)
        # self.fc2 = nn.Linear(512, 32)
        self.policy = nn.Linear(1024, num_actions)
        self.value2 = nn.Linear(32, 1)
    
    def forward(self, x):
        B, _, H, W = x.shape
        flattened_channel = x.contiguous().reshape(B, -1).long()
        embedded_channel = self.embeddings(flattened_channel)
        embedded_channel = embedded_channel.permute(0, 2, 1).contiguous().reshape(B, 16, H, W)
        x = embedded_channel

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = x.contiguous().reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.policy(x), dim=1)
        value = F.relu(self.value1(x))
        value = self.value2(x)
        return policy, value