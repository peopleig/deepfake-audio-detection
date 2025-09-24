import torch
import torch.nn as nn
import torchvision.models as tv

class ResNetAudioClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        self.backbone = tv.resnet34(weights=tv.ResNet34_Weights.DEFAULT if pretrained else None)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)
