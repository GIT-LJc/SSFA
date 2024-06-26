from .cifar import get_cifar100, get_cifar100c
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def build_dataset(args):
    if args.dataset == "cifar100":
        train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

        labeled_traindataset, unlabeled_traindataset, labeled_testdataset = get_cifar100(args)
        if args.corruption != 'none':
            unlabeled_traindataset, unlabeled_testdataset, test_unseen_dataset = get_cifar100c(args)
        else:
            unlabeled_testdataset = labeled_testdataset

        labeled_trainloader = DataLoader(
            labeled_traindataset,
            sampler=train_sampler(labeled_traindataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True)
        unlabeled_trainloader = DataLoader(
            unlabeled_traindataset,
            sampler=train_sampler(unlabeled_traindataset),
            batch_size=args.batch_size * args.mu,
            num_workers=args.num_workers,
            drop_last=True)
        labeled_testloader = DataLoader(
            labeled_testdataset,
            sampler=SequentialSampler(labeled_testdataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers)
        unlabeled_testloader = DataLoader(
            unlabeled_testdataset,
            sampler=SequentialSampler(unlabeled_testdataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers)

    return labeled_trainloader, unlabeled_trainloader, labeled_testloader, unlabeled_testloader
