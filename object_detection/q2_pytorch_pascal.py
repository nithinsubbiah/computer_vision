from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

import utils
from q0_hello_mnist import SimpleCNN
from caffe_net import CaffeNet
from voc_dataset import VOCDataset
import torchvision.models as models

from tensorboardX import SummaryWriter

from datetime import date
date_str = date.today().strftime("%d_%m_%Y")


def main():
    
    writer = SummaryWriter()

    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval')
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    # model = SimpleCNN(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=227, c_dim=3).to(device)
    # model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=227, c_dim=3).to(device)
    model = models.resnet18(pretrained=True)
    model_resnet = True
    if(model_resnet):
        model.fc = nn.Linear(in_features=512, out_features=len(VOCDataset.CLASS_NAMES), bias=True)

    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    cnt = 0
    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(device), target.to(device), wgt.to(device)           
            optimizer.zero_grad()
            output = model(data)
            criterion = torch.nn.BCEWithLogitsLoss(weight=wgt)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                writer.add_scalar('Loss/train', loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, eval_mAP = utils.eval_dataset_map(model, device, test_loader)
                model.train()
                writer.add_scalar('mAP/test', eval_mAP, cnt)
            cnt += 1

            
        scheduler.step()
        if not model.conv1.weight.grad is None:
            writer.add_histogram('conv1_histogram_of_grad', model.conv1.weight.grad.flatten().detach(), cnt)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, cnt)
        writer.add_image('train_images'+str(epoch), data[0])
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "./checkpoints/model_epoch_"+str(epoch) +"_"+date_str+".pth")

    torch.save(model.state_dict(), "./checkpoints/model_epoch_"+str(epoch) +"_"+date_str+".pth")
    
    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    AP, mAP = utils.eval_dataset_map(model, device, test_loader)

    print('----test-----')
    print(AP)
    print('mAP: ', mAP)

if __name__ == '__main__':
    args, device = utils.parse_args()
    main()