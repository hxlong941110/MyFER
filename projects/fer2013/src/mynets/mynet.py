# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

"""
修改AlexNet
"""
class MyNet(nn.Module):
	def __init__(self, num_classes=7):
		super(MyNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
		self.fc1 = nn.Linear(in_features=4 * 4 * 64, out_features=1024)
		self.fc2 = nn.Linear(in_features=1024, out_features=512)
		self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2, ceil_mode=True)
		x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2, ceil_mode=True)
		x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2, ceil_mode=True)
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training, p=0.4)
		x = F.relu(self.fc2(x))
		x = F.dropout(x, training=self.training, p=0.4)
		x = self.fc3(x)
		return x
		