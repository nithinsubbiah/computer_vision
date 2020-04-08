import argparse
import os
import shutil
import time
import sys
import math
sys.path.insert(0, 'faster_rcnn')
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import visdom
from tensorboardX import SummaryWriter

from datasets.factory import get_imdb
from custom import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--ip',
    type=str,
    help='Amazon EC2 public IP for visdom')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0
cnt = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1
    # seed = 1
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)  # Numpy module.
    # #random.seed(seed)  # Python random module.
    # torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # def _init_fn(worker_id):
    #    np.random.seed(int(seed))
	
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(False),
        # shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=torch.utils.data.SequentialSampler(train_dataset))
        # sampler=train_sampler,
        # worker_init_fn=_init_fn)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(
            test_imdb,
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
        # worker_init_fn=_init_fn)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()
    #if args.vis:
        # Update server here
    visdom_logger = visdom.Visdom(server=args.ip,port='8097')
    tboard_writer = SummaryWriter(flush_secs=1)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, visdom_logger, tboard_writer)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, visdom_logger, epoch)
            
            tboard_writer.add_scalar('eval/metric1', m1, epoch)
            tboard_writer.add_scalar('eval/metric2', m2, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, visdom_logger, tboard_writer):
    
    global cnt

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()
    
    # switch to train mode
    model.train()

    no_plotted = 0
    plot_epoch = [0,15,29,44]

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        optimizer.zero_grad()
        output = model(input_var)
        # imoutput =torch.sigmoid(output)
        if args.arch == 'localizer_alexnet':
            imoutput = torch.squeeze(F.max_pool2d(output,output.shape[2]))
        elif args.arch == 'localizer_alexnet_robust':
            imoutput = torch.squeeze(F.avg_pool2d(output,output.shape[2]))

        loss = criterion(imoutput, target_var)
        loss.backward()

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1, input.size(0))
        avg_m2.update(m2, input.size(0))

        # compute gradient and do SGD step
        optimizer.step()
        #scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        tboard_writer.add_scalar('train/loss', loss.item(), cnt)
        tboard_writer.add_scalar('train/metric1', m1, cnt)
        tboard_writer.add_scalar('train/metric2', m2, cnt)

        if epoch in plot_epoch:
            if((i+1)%75==0 and no_plotted<2):
                #plot_idx = np.random.choice(input.shape[0])
                plot_idx = 0
                gt_class = np.where(target[plot_idx]==1)[0][0]

                heatmap = output[plot_idx][gt_class].data.cpu().numpy()	    
                img_plot = input[plot_idx].data.numpy()

                img_plot = (img_plot-np.min(img_plot))*255/(np.max(img_plot)-np.min(img_plot))
                img_plot = img_plot.astype(np.uint8) 
                
                visdom_logger.image(img_plot, opts=dict(title='train/image_'+str(epoch)+'_'+str(i),store_history=True))
                visdom_logger.heatmap(heatmap, opts=dict(title='train/heatmap_'+str(epoch)+'_'+str(i)+str(gt_class),store_history=True))
                tboard_writer.add_image('train/images_'+str(epoch)+'_'+str(i), img_plot)
                heatmap = (heatmap-np.min(heatmap))*255/(np.max(heatmap)-np.min(heatmap))
                heatmap = np.expand_dims(heatmap,axis=0)
                tboard_writer.add_image('train/heatmap_'+str(epoch)+'_'+str(i), heatmap)
                
                no_plotted+=1
	cnt+=1
        # End of train()


def validate(val_loader, model, criterion, visdom_logger, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    no_plotted = 0

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``

        output = model(input_var)
        imoutput = torch.squeeze(F.max_pool2d(output,output.shape[2]))
        # imoutput = torch.sigmoid(imoutput)
        loss = criterion(imoutput, target_var)
        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1, input.size(0))
        avg_m2.update(m2, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # tboard_writer.add_scalar('eval/loss', loss.item(), cnt)

        if(no_plotted<1):
            plot_idx = np.random.choice(input.shape[0])
            gt_class = np.where(target[plot_idx]==1)[0][0]

            heatmap = output[plot_idx][gt_class].data.cpu().numpy()	    
            img_plot = input[plot_idx].data.numpy()

            img_plot = (img_plot-np.min(img_plot))*255/(np.max(img_plot)-np.min(img_plot))
            img_plot = img_plot.astype(np.uint8) 
            
            visdom_logger.image(img_plot, opts=dict(title='eval/image_'+str(epoch)+'_'+str(i),store_history=True))
            visdom_logger.heatmap(heatmap, opts=dict(title='eval/heatmap_'+str(epoch)+'_'+str(i)+str(gt_class),store_history=True))
            
            no_plotted+=1

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    output = torch.sigmoid(output)

    output = output.cpu().numpy()
    target = target.cpu().numpy()
    nclasses = target.shape[1]
    F1 = []
    for cid in range(nclasses):
        target_cls = target[:, cid].astype('float32')
        output_cls = output[:, cid].astype('float32')
        output_cls[output_cls>=0.7] = 1
        output_cls[output_cls<0.7] = 0
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        # output_cls -= 1e-5 * target_cls
        f1 = sklearn.metrics.f1_score(target_cls, output_cls,average='binary')
        if math.isnan(f1):
            f1=0
	F1.append(f1)
    metric1_score = np.mean(F1)
    ###
    return metric1_score

def metric2(output, target):
    #TODO: Ignore for now - proceed till instructed
    output = torch.sigmoid(output)
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    nclasses = target.shape[1]
    AP = []
    for cid in range(nclasses):
        target_cls = target[:, cid].astype('float32')
        output_cls = output[:, cid].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        output_cls -= 1e-5 * target_cls
        ap = sklearn.metrics.average_precision_score(
            target_cls, output_cls)
        if math.isnan(ap):
	    ap=0
	AP.append(ap)
    metric2_score = np.mean(AP)
    
    return metric2_score


if __name__ == '__main__':
    main()
