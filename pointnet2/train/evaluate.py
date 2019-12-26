from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import os
import argparse

from pointnet2.models import Pointnet2ClsSSG as Pointnet
from pointnet2.data import ModelNet40Cls
import pointnet2.data.data_utils as d_utils
from pointnet2.utils import pointnet2_utils

import time
import shutil
import numpy as np

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for evaluate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "-num_points", type=int, default=1024, help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=float, default=20, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-checkpoint", type=str, default='checkpoints/pointnet2_cls_best', help="Checkpoint to start from"
    )
    parser.add_argument(
        "-epochs", type=int, default=200, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-log_file", type=str, default='./log/train_log.txt', help="log information"
    )
    parser.add_argument('-print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 100)')

    return parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count
                            
def evaluate(val_loader, model, criterion, votes):
    # switch to evaluate mode
#    PointcloudScale = d_utils.PointcloudScale()
    model.eval()
    global_preds =[]
    for rep in range(1):
        losses     = AverageMeter()
        top1       = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
            preds = []
            input      = input.cuda()
            target     = target.cuda()
            input_var  = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            target_var = target_var.view(-1)
        
            for vote in range(votes):
                PointcloudRotatebyVote = d_utils.PointcloudRotatebyAngle(vote / votes * 2 * np.pi)
                input_var=PointcloudRotatebyVote(input_var)
                # compute output
                with torch.no_grad():
                    output = model(input_var)
                loss   = criterion(output, target_var)
                # measure accuracy and record loss
                _, classes = torch.max(output, -1)
                acc = (classes == target_var).float().sum() / target_var.numel()
                preds.append(acc)
            top1_acc = max(preds)
            losses.update(loss.item(), input.size(0))
            top1.update(top1_acc, input.size(0))
        print('Repeat{:d}\t,Acc{:.6f}'.format(rep, top1.avg))
        global_preds.append(top1.avg)
    best_acc = max(global_preds)
    print('\nBest voting acc: {:.6f}'.format(best_acc))
    return best_acc
            
lr_clip = 1e-5
bnm_clip = 1e-2

def checkpoint_state(model=None,
                     optimizer=None,
                     best_prec=None,
                     epoch=None,
                     it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        'epoch': epoch,
        'it': it,
        'best_prec': best_prec,
        'model_state': model_state,
        'optimizer_state': optim_state
    }
    
def save_checkpoint(state,
                    is_best,
                    filename='checkpoint',
                    bestname='model_best'):
    filename = '{}.pth.tar'.format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth.tar'.format(bestname))

if __name__ == "__main__":
    args = parse_args()
   
    model = Pointnet(input_channels=0, num_classes=40, use_xyz=True)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    if args.checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
        model, filename=args.checkpoint.split(".")[0]#, optimizer
    )
    if checkpoint_status is not None:
        it, start_epoch, best_loss = checkpoint_status


    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    best_top1 = 0
    start_epoch = 1

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status
    votes = 12
    transform = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
#                d_utils.PointcloudRotateByAngle(i / votes * 2 * np.pi),
        ]
    )

    test_set = ModelNet40Cls(args.num_points, transforms=transform, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    top1 = evaluate(test_loader, model, criterion, votes)

