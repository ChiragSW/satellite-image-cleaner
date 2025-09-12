import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# classify cloud and clear images
class CloudClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CloudClassifier, self).__init__()
        
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.base_model.fc.in_features
        # replace the final fully connected layer for classification task
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
