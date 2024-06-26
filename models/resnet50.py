import torch
import torch.nn as nn
from torchvision import models
import os

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, channels, momentum=1e-3, eps=1e-3):
        super().__init__(channels)
        self.update_batch_stats = True

    def forward(self, x):
        if self.update_batch_stats:
            return super().forward(x)
        else:
            return nn.functional.batch_norm(
                x, None, None, self.weight, self.bias, True, self.momentum, self.eps
            )

def relu():
    return nn.LeakyReLU(0.1)


class ResNet50Fc(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self, num_classes, model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.resnet50(pretrained=False)
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_resnet = models.resnet50(pretrained=True)

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features
        
        self.bottleneck = nn.Linear(self.__in_features, 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        from torch.cuda.amp import autocast as autocast
        with autocast():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.bottleneck(x)
            c = self.fc(x)
        return c

    def output_num(self):
        return self.__in_features