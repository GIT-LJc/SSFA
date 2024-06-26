import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import random
from collections import defaultdict

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

common_corruptions_15 = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog','brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
common_corruptions_10 = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'glass_blur','motion_blur', 'frost', 'fog', 'elastic_transform', 'pixelate', 'jpeg_compression']
common_corruptions_5 = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'glass_blur', 'pixelate']

common_corruptions = {15:common_corruptions_15, 10:common_corruptions_10, 5:common_corruptions_5}
unseen_corruptions = defaultdict(list)
for k,v in common_corruptions.items():
    unseen_corruptions[k] = list(set(common_corruptions_15) - set(v))


def prepare_transforms(dataset):

    normalize = transforms.Normalize(mean=cifar100_mean, std=cifar100_std)

    te_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    tr_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        normalize])

    return tr_transforms, te_transforms



def get_cifar100(args, root='./data'):

    transform_labeled, transform_val = prepare_transforms(args.dataset)

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def get_cifar100c(args, root0='./data'):
    print('prepare mixed unlabeled dataset...')   # Select mixed unlabeled samples according to ratio
    label_size = 50000
    tesize = int(args.ratio * label_size)
    sample = []
    if args.corruption == 'mix15':
        num_mix = 15
    elif args.corruption == 'mix10':
        num_mix = 10
    elif args.corruption == 'mix5':
        num_mix = 5
    else:
        num_mix = 1
        unseen_corruptions[1] = list(set(common_corruptions_15) - set(args.corruption))

    root = root0 + '/CIFAR-100-C-train'
    data, label, sample = prepare_mix_corruption(args, tesize, num_mix, root, common_corruptions, args.corruption) 
    
    if label_size - tesize:
        random.seed(args.seed)
        idxs = range((args.corruption_level-1)*label_size, args.corruption_level*label_size)

        unlabeled_idxs = list(set(idxs)-set(sample))
        unlabeled_idxs = np.array(unlabeled_idxs)-(args.corruption_level-1)*label_size
        base_dataset = datasets.CIFAR100(root, train=True, download=True)
        unlabeled_data = base_dataset.data[unlabeled_idxs, :]
        unlabeled_label = np.array(base_dataset.targets)[unlabeled_idxs]
        
        # mix clean and corrupted
        data = np.concatenate([data, unlabeled_data])
        label = np.concatenate([label, unlabeled_label])

    train_unlabeled_dataset = CIFAR_C_SSL(root='./data', data=data, label=label, train=True,transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))


    _, te_transforms = prepare_transforms(args.dataset)
    print('prepare cifar100-c-test dataset...')
    
    # tesize = 10000
    test_dataset = datasets.CIFAR100(
        root0, train=False, transform=te_transforms, download=False)

    tesize = len(test_dataset.targets)
    root = root0 + "/CIFAR-100-C"
    data, label, _ = prepare_mix_corruption(args, tesize, num_mix, root, common_corruptions, args.corruption) 
    
    test_corrupted_dataset = CIFAR_C_SSL(root='./data', data=data, label=label, train=False, transform=te_transforms, download=False)

    data, label, _ = prepare_mix_corruption(args, tesize, num_mix, root, unseen_corruptions, args.corruption) 
    test_unseen_dataset = CIFAR_C_SSL(root='./data', data=data, label=label, train=False,  transform=te_transforms, download=False)

    return train_unlabeled_dataset, test_corrupted_dataset, test_unseen_dataset



def get_single_corp(args, root, corruption, num, label_list):
    trset_raw = np.load(root + '/%s.npy' %(corruption))
    labels = np.load(root + '/labels.npy')
    label_size = int(len(labels)/5)
    if args.corruption_level:
        idxs = range((args.corruption_level-1)*label_size, args.corruption_level*label_size)
    else:
        idxs =  range(len(labels))
    random.seed(args.seed)
    if len(label_list):   # Selecting samples without repetition
        idxs = list(set(idxs)-set(label_list))
    sample_list = random.sample(idxs, num) 
    data = trset_raw[sample_list,:]
    label = labels[sample_list]
    label_list = label_list + sample_list
    return data, label, label_list


def prepare_mix_corruption(args, tesize, num_mix, root, corruptions, corp=None):
    teset_raw = []
    telabel_raw = []
    telabel_sample = []

    if num_mix == 1:
        teset_mix_raw, telabel_mix_raw, _ = get_single_corp(args, root, corp, tesize, telabel_raw)
        return teset_mix_raw, telabel_mix_raw 
    
    else:
        num_single = int(tesize / num_mix)
        for corruption in corruptions[num_mix]:
            print('prepare corruption of ' + corruption)
            data, label, telabel_sample = get_single_corp(args, root, corruption, num_single, telabel_sample)
            
            if len(teset_raw):
                teset_raw = np.concatenate([teset_raw, data])
                telabel_raw = np.concatenate([telabel_raw, label])
            else:
                teset_raw = data
                telabel_raw = label
                
        return teset_raw, telabel_raw, telabel_sample
    


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CIFAR_C_SSL(datasets.CIFAR100):
    def __init__(self, root, data, label, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.data = data
        self.targets = np.array(label)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
