
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck
from torch import nn
import math
import torch.utils.model_zoo as model_zoo
import torch


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class MultiTaskModel(nn.Module):

    def __init__(self, model1,model2, task1_classes=1000,task2_classes=1000,channels = 10):
        self.inplanes = 64
        super(MultiTaskModel, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.model1 = model1
        self.model1.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model1.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model1.fc = nn.Linear(2048, task1_classes)
        self.model2 = model2
        self.model2.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model2.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model2.fc = nn.Linear(2048, task2_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.model1(x)
        out2 = self.model2(x)

        return out1, out2


def multi_model( model1,model2,**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MultiTaskModel( model1,model2,**kwargs)

    return model



