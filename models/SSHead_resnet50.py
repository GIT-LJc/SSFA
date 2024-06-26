import torch
from torch import nn
import math
import copy
from models.resnet50 import ResNet50Fc


class ViewFlatten(nn.Module):
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ExtractorHead(nn.Module):
    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        from torch.cuda.amp import autocast as autocast
        with autocast():
            return self.head(self.ext(x))

def extractor_from_layer4(net):
    layers = [net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool, ViewFlatten()]
    return nn.Sequential(*layers)

def head_on_layer4(net, classes=4):
    head = copy.deepcopy([net.bottleneck])
    head.append(nn.Linear(256, classes))
    return nn.Sequential(*head)

def extractor_from_layer3(net):
    layers = [net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3]
    return nn.Sequential(*layers)

def head_on_layer3(net, classes=4):
    head = copy.deepcopy([net.layer4, net.avgpool, ViewFlatten(), net.bottleneck])
    head.append(nn.Linear(256, classes))
    return nn.Sequential(*head)

def extractor_from_layer2(net):
    layers = [net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2]
    return nn.Sequential(*layers)

def head_on_layer2(net, classes=4):
    head = copy.deepcopy([net.layer3, net.layer4, net.avgpool, ViewFlatten(), net.bottleneck])
    head.append(nn.Linear(256, classes))
    return nn.Sequential(*head)


def build_ssl_resnet50(args, net, ssh):
    head = copy.deepcopy(ssh.head)
    if args.shared == 'none':
        args.shared = None

    if args.shared == 'layer4' or args.shared is None:
        ext = extractor_from_layer4(net)
    if args.shared == 'layer3':
        ext = extractor_from_layer3(net)
    elif args.shared == 'layer2':
        ext = extractor_from_layer2(net)
        
    newssh = ExtractorHead(ext, head)
    return ext, head, newssh


def build_model_resnet50(args):
    net = ResNet50Fc(num_classes=args.num_classes, model_path = '/home/liangjiachen/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')
    # net = ResNet50Fc(num_classes=args.num_classes)

    print('Building model...')
    
    if args.shared == 'none' :
        args.shared = None

    if args.ssl == 'simclr':
        classes = 128
    else:
        classes = 4

    if args.shared == 'layer4' or args.shared is None:
        ext = extractor_from_layer4(net)
        head = head_on_layer4(net, classes)

    elif args.shared == 'layer3' or args.shared is None:
        ext = extractor_from_layer3(net)
        head = head_on_layer3(net, classes)

    else:
        ext = extractor_from_layer2(net)
        head = head_on_layer2(net, classes)

    ssh = ExtractorHead(ext, head)

    return net, ext, head, ssh

