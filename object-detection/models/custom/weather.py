import torch
import torch.nn as nn

class Weather(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.layer1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), #1 * 64 * 64 -> 16 * 64 * 64
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(2), # 16 * 14 * 14
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # 16 * 32 * 32 -> 32 * 32 * 32
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2), # 32 * 16 * 16
    )
    self.fc = nn.Linear(8192, 4)
    
  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x