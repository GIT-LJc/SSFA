from .SSHead_resnet50 import build_model_resnet50, build_ssl_resnet50
from .SSHead_wrn import build_model_wrn, build_ssl_wrn


def build_model(args):
    if args.arch == 'wideresnet':
        return build_model_wrn(args)
    elif args.arch == 'resnet50':
        return build_model_resnet50(args)


def build_ssl(args, net, ssl):
    if args.arch == 'wideresnet':
        return build_ssl_wrn(args, net, ssl)
    elif args.arch == 'resnet50':
        return build_ssl_resnet50(args, net, ssl)


           