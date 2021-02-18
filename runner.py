from types import SimpleNamespace

import os
import time
import util
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from sklearn.metrics import confusion_matrix, f1_score

import backbone

import config
from config import DEVICE as device

# print(model_names)

args = SimpleNamespace()

# DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


args.device_name = config.DEVICE

args.epochs = 200
args.start_epoch = 0
args.save_every = 1

args.print_freq = 10

args.save_dir = "vanilla_224_no_dropout"
args.pretrained = True
args.evaluate = False

args.lr = 1e-4
# args.momentum = 0.9
# args.weight_decay = 0

best_acc = 0

def main():
    global args, best_acc

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_loader, val_loader = util.get_cifar10_dataloaders()

    model = backbone.SqueezeNet(type='vanilla', input_shape=32, num_classes=config.NUMBER_OF_CLASSES)
    model.to(device)
    model = nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, eps=1e-6, verbose=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        util.printf("train_loss: %.3f, train_acc: %.3f\n", train_loss, train_acc / 100)

        # evaluate on validation set
        val_acc, val_loss, cm, f1 = validate(val_loader, model, criterion)
        util.printf("val_loss: %.3f, val_acc: %.3f\n", val_loss, val_acc / 100)

        # remember best prec@1 and save vanilla_224_no_dropout
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if is_best:
            checkpoint_filename = f'epoch_{epoch}_trainAcc_{train_acc:.3f}_valAcc_{val_acc:.3f}_checkpoint.th'
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, is_best, filename=os.path.join(args.save_dir, checkpoint_filename))

        lr_scheduler.step(val_loss)
        # lr_scheduler.step()

        del train_acc, train_loss, val_acc, val_loss

    model_checkpoints = [f for f in os.listdir(args.save_dir) if os.path.isfile(os.path.join(args.save_dir, f))]
    state_dict = torch.load(os.path.join(args.save_dir, model_checkpoints[-1]))["state_dict"]
    model.load_state_dict(state_dict)

    pred = torch.empty(0, config.NUMBER_OF_CLASSES)
    y_val = torch.empty(0, config.NUMBER_OF_CLASSES)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = input.to(device)

            # compute output
            output = model(input_var)
            pred = torch.cat((pred, output.cpu()), dim=0)

            y_val = torch.cat((y_val, target), dim=0)

    pred = pred.numpy()

    y_pred = np.argmax(pred, axis=1)
    y_true = y_val
    cm = confusion_matrix(y_true, y_pred)

    print(cm)
    print(f1_score(y_true, y_pred, average=None))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = input.to(device)
        target_var = target.to(device)

        optimizer.zero_grad()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step

        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input_var.size(0))
        top1.update(prec1.item(), input_var.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i>0 and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    return top1.avg, losses.avg

def validate(val_loader, model, criterion, print_cm = True):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode

    end = time.time()

    model.eval()

    pred = torch.empty(0, config.NUMBER_OF_CLASSES)
    y_test = torch.empty(0)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input_var = input.to(device)
            target_var = target.to(device)

            # compute output
            output = model(input_var)
            pred = torch.cat((pred, output.cpu()), dim=0)
            y_test = torch.cat((y_test, target_var.cpu()), dim=0)

            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input_var.size(0))
            top1.update(prec1.item(), input_var.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i != 0 and i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    # print(' * Prec@1 {top1.avg:.3f}'
    #       .format(top1=top1))

    # if print_cm:
    pred = pred.numpy()
    #
    pred_y = np.argmax(pred, axis=1)
    #     cm = confusion_matrix(y_test, pred_y)
    #
    #     print(cm)
    #     print(f1_score(y_test, pred_y, average=None))

    return top1.avg, losses.avg, confusion_matrix(y_test, pred_y), f1_score(y_test, pred_y, average=None)

def save_checkpoint(state, is_best, filename='vanilla_224_no_dropout.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
