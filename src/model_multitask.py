# src/model_multitask.py
import torch.nn as nn
from torchvision import models

class MultiTask(nn.Module):
    def __init__(self, n_types=5, n_grades=3):  # adjust numbers
        super().__init__()
        self.backbone=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        d=self.backbone.fc.in_features
        self.backbone.fc=nn.Identity()
        self.head_type = nn.Linear(d, n_types)
        self.head_grade= nn.Linear(d, n_grades)  # or 1 for regression
    def forward(self,x):
        f=self.backbone(x)
        return self.head_type(f), self.head_grade(f)
