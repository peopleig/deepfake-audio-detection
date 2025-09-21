import torch
import torch.nn as nn
import torchvision.models as models

class ResNetAudioClassifier(nn.Module):
    def __init__(self, n_input_channels=1, n_classes=1, pretrained=False):
        super(ResNetAudioClassifier, self).__init__()
        self.resnet = models.resnet34(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(
            n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))

if __name__ == "__main__":
    model = ResNetAudioClassifier()
    dummy_input = torch.randn(2, 1, 128, 400) 
    out = model(dummy_input)
    print(out.shape)
