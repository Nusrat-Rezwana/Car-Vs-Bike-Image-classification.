import torch.nn as nn
import torchvision.models as models


model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# print(model)
class CarBikeClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(CarBikeClassifier, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)



