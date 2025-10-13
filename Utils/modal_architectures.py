# model_architectures.py

import torch
import torch.nn as nn
from torchvision import models
import config

class HybridCNN(nn.Module):
    """A hybrid model combining ResNet50 and DenseNet121."""
    def __init__(self, num_classes, pretrained=True, freeze_backbones=True):
        super().__init__()
        r_weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        d_weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None

        # Backbone A: ResNet50
        self.resnet = models.resnet50(weights=r_weights)
        self.resnet.fc = nn.Identity()

        # Backbone B: DenseNet121
        self.densenet = models.densenet121(weights=d_weights)
        self.densenet.classifier = nn.Identity()

        feat_dim = 2048 + 1024 # ResNet50 output + DenseNet121 output

        if freeze_backbones:
            for m in [self.resnet, self.densenet]:
                for param in m.parameters():
                    param.requires_grad = False

        # Fusion + Classifier Head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, config.HIDDEN_UNITS),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_UNITS, num_classes)
        )

    def forward(self, x):
        f1 = self.resnet(x)
        f2 = self.densenet(x)
        fused = torch.cat([f1, f2], dim=1)
        return self.head(fused)

def build_model(num_classes):
    """Builds and returns the HybridCNN model."""
    model = HybridCNN(num_classes=num_classes, pretrained=True, freeze_backbones=True)
    print("HybridCNN model created.")
    return model