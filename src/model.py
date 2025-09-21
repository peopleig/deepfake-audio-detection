import torch
import torch.nn as nn
import torchvision.models as models

class ResNetAudioClassifier(nn.Module):
    def __init__(self, n_input_channels=1, n_classes=1, pretrained=False):
        """
        Args:
            n_input_channels: 1 for mono audio features (e.g., LFCC/Mel)
            n_classes: 1 for binary classification (bonafide vs spoof)
            pretrained: whether to use ImageNet pretrained ResNet
        """
        super(ResNetAudioClassifier, self).__init__()

        # Load ResNet34
        self.resnet = models.resnet34(pretrained=pretrained)

        # Modify first conv layer to accept n_input_channels
        self.resnet.conv1 = nn.Conv2d(
            n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify final layer for binary classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, n_features, T)
        """
        return torch.sigmoid(self.resnet(x))  # Output probability for binary classification


# Example usage
if __name__ == "__main__":
    model = ResNetAudioClassifier()
    dummy_input = torch.randn(2, 1, 128, 400)  # batch_size=2, 128 mel bins, 400 time frames
    out = model(dummy_input)
    print(out.shape)  # Should print torch.Size([2, 1])
