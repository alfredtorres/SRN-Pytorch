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
from pointnet2.models.pointnet2_msg_cls import model_fn_decorator
from pointnet2.data import ModelNet40Cls
import pointnet2.data.data_utils as d_utils

import time
import shutil
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
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
        "-checkpoint", type=str, default=None, help="Checkpoint to start from"
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


lr_clip = 1e-5
bnm_clip = 1e-2

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
        
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()

    model.train()

    end = time.time()    
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # compute output
        input      = input.cuda()
        target     = target.cuda()
        input_var  = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var = target_var.view(-1)
        
        output = model(input_var)
        
        loss   = criterion(output, target_var)
        
        optimizer.zero_grad()
        # measure accuracy and record loss
        #_, loss, eval_res = criterion(model, batch)
        _, classes = torch.max(output, -1)
        acc = (classes == target_var).float().sum() / target_var.numel()
        losses.update(loss.item(), input.size(0))
        top1.update(acc, input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
       

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            
            f.writelines('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\n'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    losses     = AverageMeter()
    top1       = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        input      = input.cuda()
        target     = target.cuda()
        input_var  = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var = target_var.view(-1)
        # compute output
        with torch.no_grad():
            output = model(input_var)
        
        loss   = criterion(output, target_var)
        # measure accuracy and record loss
        _, classes = torch.max(output, -1)
        acc = (classes == target_var).float().sum() / target_var.numel()
        losses.update(loss.item(), input.size(0))
        top1.update(acc, input.size(0))


    print('\nTest set: Average loss: {}, Accuracy: ({})\n'.format(losses.avg, top1.avg))
    f.writelines('\nTest set: Average loss: {}, Accuracy: ({})\n'.format(losses.avg, top1.avg))

    return top1.avg

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
    if not osp.isdir("log"):
        os.makedirs("log")
    f = open(args.log_file, 'w')

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale(),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
            d_utils.PointcloudRandomInputDropout(),
        ]
    )

    test_set = ModelNet40Cls(args.num_points, transforms=transforms, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    train_set = ModelNet40Cls(args.num_points, transforms=transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = Pointnet(input_channels=0, num_classes=40, use_xyz=True)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    lr_lbmd = lambda epoch: max(
        args.lr_decay ** (int(epoch // args.decay_step)),
        lr_clip / args.lr,
    )
    bn_lbmd = lambda epoch: max(
        args.bn_momentum
        * args.bnm_decay ** (int(epoch // args.decay_step)),
        bnm_clip,
    )

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

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=it)
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bn_lbmd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`
    
    if not osp.isdir("checkpoints"):
        os.makedirs("checkpoints")
    checkpoint_name="checkpoints/pointnet2_cls"
    best_name="checkpoints/pointnet2_cls_best"

    for epoch in range(args.epochs):      
        #lr_f.write()
#        adjust_learning_rate(optimizer, epoch)
        print('lr is :',(optimizer.param_groups[0]['lr']))
        # train for one epoch
        train(train_loader, model,  criterion, optimizer, epoch)
        # evaluate on validation set
        top1 = validate(test_loader, model,  criterion)    
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        if bnm_scheduler is not None:
            bnm_scheduler.step(epoch-1)
        # save the learned parameters
        # save the learned parameters
        is_best = top1 > best_top1
        best_top1 = max(best_top1, top1)
        save_checkpoint(
            checkpoint_state(model, optimizer,
                             top1, args.epochs, epoch),
            is_best,
            filename=checkpoint_name,
            bestname=best_name)
    ## rewrite the training process end
    f.close()
