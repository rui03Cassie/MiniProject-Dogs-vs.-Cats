from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models


class CNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # output: [B, 512, 1, 1]
            )  
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(dropout), 
            nn.Linear(512, 256), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout), 
            nn.Linear(256, num_classes),
            )  
          
        for ele in self.modules():
            if isinstance(ele, nn.Conv2d):
                nn.init.kaiming_normal_(ele.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(ele, nn.BatchNorm2d):
                nn.init.ones_(ele.weight)
                nn.init.zeros_(ele.bias)
            elif isinstance(ele, nn.Linear):
                nn.init.normal_(ele.weight, mean=0, std=0.01)
                nn.init.zeros_(ele.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.classifier(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2, pretrained=True, train_backbone=False, unfreeze_layer4=True):
        super().__init__()
        model = models.resnet18(pretrained=pretrained)
        # 不能开train_backbone的同时开layer4
        if train_backbone and unfreeze_layer4:
            raise ValueError("Choose either train_backbone or unfreeze_layer4, not both.")
        
        for param in model.parameters():
            param.requires_grad = False
        
        if train_backbone:
            for param in model.parameters():
                param.requires_grad = True
        elif unfreeze_layer4:
            for param in model.layer4.parameters():
                param.requires_grad = True
        
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        for param in model.fc.parameters():
            param.requires_grad = True
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

def build_model(model_name, num_classes, dropout=0.3, pretrained=False, train_backbone=False, unfreeze_layer4=False):
    model_name = model_name.lower()
    if model_name == "cnn":
        return CNN(num_classes, dropout)
    elif model_name == "resnet":
        return ResNet(num_classes, dropout, pretrained, train_backbone, unfreeze_layer4)
    raise ValueError(f"Unknown model: {model_name}")


def main():
    cnn = build_model("CNN")
    resNet = build_model("ResNet")

    print("CNN total params:", sum(p.numel() for p in cnn.parameters()))
    print("CNN trainable params:", sum(p.numel() for p in cnn.parameters() if p.requires_grad))
    print("ResNet total params:", sum(p.numel() for p in resNet.parameters()))
    print("ResNet trainable params:", sum(p.numel() for p in resNet.parameters() if p.requires_grad))

if __name__ == "__main__":
    main()