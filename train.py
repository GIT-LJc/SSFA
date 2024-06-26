import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.build_dataset import build_dataset
from utils import AverageMeter, accuracy

from models.build_model import build_model, build_ssl
from helpers.rotation import rotate_batch
from helpers.ema import update_ema
from helpers.entropy import collect_params
import copy
import json
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

logger = logging.getLogger(__name__)
best_acc = 0

def ssfa_optimized(args, model):
    if args.ssl == 'entropy':
        return collect_params(model)
    elif args.ssl == 'rotation' or args.ssl == 'simclr':
        if args.optim_params == 'all':
            return model.parameters()
        return model.ext.parameters()

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)



def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu_id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar100', 'office31', 'officehome'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--num-unlabeled', type=int, default=4000,
                        help='number of unlabeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext', 'resnet50'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=102400, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lambda-ssl', default=1, type=float,
                        help='coefficient of ssl loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    # add ssl parameters
    parser.add_argument('--update', action="store_true", help="use ssfa")
    parser.add_argument('--optim_params', default='all', help="the parameters to be optimzied", choices=['all', 'shared'])
    parser.add_argument('--shared', default='layer2', help="the number of shared layers")
    parser.add_argument('--rotation_type', default='rand')
    parser.add_argument('--ssl', type=str, default='none', choices=['rotation', 'simclr', 'entropy', 'none'], help="use which ssl task")
    parser.add_argument('--corruption', default='none')
    parser.add_argument('--corruption_level', default=5, type=int)
    parser.add_argument('--ratio', default=0.2, type=float, help="the ratio of corrupted unlabeled images")
    parser.add_argument('--src-domain', default='Art', type=str, choices=["Art", "Clipart", "Product", "RealWorld", "amazon", "dslr", "webcam"])
    parser.add_argument('--tgt-domain', default='other', type=str, choices=["Art", "Clipart", "Product", "RealWorld", "other", "all", "amazon", "dslr", "webcam"])

    args = parser.parse_args()
    global best_acc

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.arch == 'wideresnet':
        args.model_depth = 28
        args.model_width = 8
    elif args.arch == 'resnext':
        args.model_cardinality = 8
        args.model_depth = 29
        args.model_width = 64
    
    if args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'officehome' :
        args.num_classes = 65
        args.data = './data/OfficeHome'
    elif args.dataset == 'office31' :
        args.num_classes = 31
        args.data = './data/OFFICE31'

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    
    labeled_trainloader, unlabeled_trainloader, test_loader, cortest_loader = build_dataset(args)

    net, ext, head, ssl = build_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    net.to(args.device)
    ssl.to(args.device)
    head.to(args.device)

    no_decay = ['bias', 'bn']
 
    if args.ssl is not 'none':
        grouped_parameters = [
            {'params': [p for n, p in net.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
            {'params': [p for n, p in net.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in head.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
            {'params': [p for n, p in head.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        grouped_parameters = [
            {'params': [p for n, p in net.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
            {'params': [p for n, p in net.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, net, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        if args.shared:
            head.load_state_dict(checkpoint['state_dict_head'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        test_model = copy.deepcopy(net)
        test_model.load_state_dict(checkpoint['ema_state_dict'])

        _, test_acc = test(args, test_loader, test_model, args.start_epoch)
        _, corptest_acc = test(args, cortest_loader, test_model, args.start_epoch)

        from helpers.writer import write_acc
        write_acc(args, test_acc, corptest_acc, args.out)
        del test_model


    if args.amp:
        from apex import amp
        net, optimizer = amp.initialize(
            net, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        # add head
        head = torch.nn.parallel.DistributedDataParallel(
            head, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)


    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")
    logger.info(f"  start epochs = {args.start_epoch}")

    net.zero_grad()
    ssl.zero_grad()

    train(args, labeled_trainloader, unlabeled_trainloader, test_loader, cortest_loader, net, ssl, head, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, cortest_loader, model, ssl, head, optimizer, ema_model, scheduler):
    global best_acc
    test_accs = []
    end = time.time()


    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
        

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    
    if args.ssl == 'simclr':
        from helpers.simclr_losses import ContrastiveLoss
        criterion_ss = ContrastiveLoss(args.batch_size * args.mu).cuda()
    

    model.train()
    ssl.train()
    no_decay = ['bias', 'bn']

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_ssl = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                # inputs_x, targets_x = labeled_iter.next()
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            if args.ssl == 'simclr':
                inputs_s = interleave(
                    torch.cat((inputs_u_w, inputs_u_s)), 2 * args.mu).to(args.device)


            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_s)), args.mu + 1).to(args.device)
            inputs_ssl = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            inputs_u_w =  inputs_u_w.to(args.device)

            targets_x = targets_x.to(args.device)
            
            # Feature adaptation 
            if args.update:
                newnet = copy.deepcopy(model)
                newnet.zero_grad()
                if args.ssl in ['rotation', 'simclr']:
                    _, _, newssl = build_ssl(args, newnet, ssl)
                else:
                    newssl = newnet
                newssl.train()
                newssl.zero_grad()
                ssfa_params = ssfa_optimized(args, newssl)

                optimizer_apt = optim.SGD(ssfa_params, lr=scheduler.get_last_lr()[0],
                                    momentum=0.9, nesterov=args.nesterov)
            
            
            if args.update:
                with autocast():
                    logits = model(inputs)
                    logits = de_interleave(logits, args.mu + 1)
                    logits_x = logits[:batch_size]
                    logits_u_s = logits[batch_size:]
                    del logits
                
                    if args.ssl == 'rotation':
                        inputs_apt, labels_apt = rotate_batch(inputs_u_w, args.rotation_type)
                        inputs_apt, labels_apt = inputs_apt.cuda(args.device), labels_apt.cuda(args.device)
                        outputs_apt = newssl(inputs_apt)
                        newloss_apt = F.cross_entropy(outputs_apt, labels_apt, reduction='mean')
                    elif args.ssl == 'simclr':
                        features = newssl(inputs_s)
                        features = de_interleave(features, 2 * args.mu) 
                        f1, f2 = features.chunk(2)
                        bsz = f1.shape[0]
                        newloss_apt = criterion_ss(f1, f2, bsz)
                    else:
                        outputs_apt = newnet(inputs_u_w)
                        newloss_apt = softmax_entropy(outputs_apt).mean(0)
                    del outputs_apt
        
                scaler.scale(newloss_apt).backward()
                scaler.step(optimizer_apt)
                scaler.update()
                del newloss_apt

            with autocast():
                if args.update:
                    logits_u_w = newnet(inputs_u_w)
                    del newnet
                    del newssl
                else:
                    logits = model(inputs_ssl)
                    logits = de_interleave(logits, 2 * args.mu + 1)
                    logits_x = logits[:batch_size]
                    logits_u_w, logits_u_s = logits[batch_size:].chunk(2)


                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                loss = Lx 
                losses_x.update(Lx.item())

                pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                Lu = (F.cross_entropy(logits_u_s, targets_u,
                                    reduction='none') * mask).mean()
                loss += args.lambda_u * Lu
                
                loss_ssl = torch.tensor(0)
                if args.ssl == 'rotation':
                    inputs_ssl, labels_ssl = rotate_batch(inputs_ssl, args.rotation_type)
                    inputs_ssl, labels_ssl = inputs_ssl.cuda(args.device), labels_ssl.cuda(args.device)
                    outputs_ssl = ssl(inputs_ssl)
                    loss_ssl = F.cross_entropy(outputs_ssl, labels_ssl, reduction='mean')
                elif args.ssl == 'simclr':
                    features = ssl(inputs_s)
                    features = de_interleave(features, 2 * args.mu) 
                    f1, f2 = features.chunk(2)
                    bsz = f1.shape[0]
                    loss_ssl = criterion_ss(f1, f2)

                loss += args.lambda_ssl * loss_ssl

            losses.update(loss.item())
            losses_ssl.update(loss_ssl.item())
            losses_u.update(Lu.item())


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            ssl.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_ssl: {loss_ssl:.4f}. Mask: {mask:.2f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        loss_ssl=losses_ssl.avg,
                        mask=mask_probs.avg))
                p_bar.update()
    
        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)
         
            corptest_loss, corptest_acc = test(args, cortest_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_ssl', losses_ssl.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            args.writer.add_scalar('test/3.corptest_acc', corptest_acc, epoch)
            args.writer.add_scalar('test/4.corptest_loss', corptest_loss, epoch)
            
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            head_to_save = head.module if hasattr(head, "module") else head

            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            filename='checkpoint.pth.tar'


            save_checkpoint({
                'epoch': epoch + 1,
                'net': model.state_dict(), 'head': head.state_dict(),
                'state_dict': model_to_save.state_dict(), 'state_dict_head': head_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out, filename)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            # write json
            acc0 = {}
            acc0["epoch"] = epoch
            acc0["source_acc"] = test_acc
            acc0["target_acc"] = corptest_acc
            acc0["source_loss"] = test_loss
            acc0["target_loss"] = corptest_loss
            acc0['mask_probs.avg'] = mask_probs.avg

            exp_name = args.dataset + '_result'
            if not os.path.exists(args.out):
                os.mkdir(args.out)
            with open(os.path.join(args.out, exp_name + ".json"), "a") as f:
                json.dump(acc0, f)
                f.write('\n')

    if args.local_rank in [-1, 0]:
        args.writer.close()
        f.close()

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device, dtype = torch.long)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        if not args.no_progress:
            test_loader.close()

    logger.info("epoch: {}".format(epoch))
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
