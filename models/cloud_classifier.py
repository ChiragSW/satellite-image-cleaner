import torch
import torch.nn as nn
import torchvision.models as models

class CloudClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CloudClassifier, self).__init__()
        
        # Pretrained ResNet backbone
        self.base_model = models.resnet18(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
