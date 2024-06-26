from torch import nn
import torch
import copy
import models.wideresnet as wrn

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

def extractor_from_layer3(net):
    layers = [net.conv1, net.block1, net.block2, net.block3, net.bn1, net.relu, nn.AdaptiveAvgPool2d(1), ViewFlatten()]
    return nn.Sequential(*layers)


def extractor_from_layer2(net):
    layers = [net.conv1, net.block1, net.block2]
    return nn.Sequential(*layers)

def head_on_layer2(net, width, classes):
    head = copy.deepcopy([net.block3, net.bn1, net.relu, nn.AdaptiveAvgPool2d(1)])
    head.append(ViewFlatten())
    head.append(nn.Linear(64 * width, classes))
    return nn.Sequential(*head)


def build_ssl_wrn(args, net, ssl):
    head = copy.deepcopy(ssl.head)
    if args.shared == 'none':
        args.shared = None
    if args.shared == 'layer3' or args.shared is None:
        ext = extractor_from_layer3(net)
    elif args.shared == 'layer2':
        ext = extractor_from_layer2(net)
        
    newssl = ExtractorHead(ext, head)
    return ext, head, newssl


def build_model_wrn(args):
    net = wrn.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes, args=args)

    print('Building model...')
    
    if args.shared == 'none' :
        args.shared = None

    if args.ssl == 'simclr':
        classes = 128
    else:
        classes = 4

    if args.shared == 'layer3' or args.shared is None:
        ext = extractor_from_layer3(net)
        head = nn.Linear(64 * args.model_width, classes)

    else:
        ext = extractor_from_layer2(net)
        head = head_on_layer2(net, args.model_width, classes)

    ssl = ExtractorHead(ext, head).cuda()

    return net, ext, head, ssl

