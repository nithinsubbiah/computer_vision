import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CaffeNet(nn.Module):

    def __init__(self, num_classes=20, inp_size=227, c_dim=3):
        super().__init__()
        self.num_classes = num_classes

        self.nonlinear = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop_out = nn.Dropout(p=0.5) 

        self.conv1 = nn.Conv2d(in_channels=c_dim, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(9216, 4096)
        self.fc1 = nn.Sequential(*get_fc(9216, 4096, 'relu'))
        self.fc2 = nn.Sequential(*get_fc(4096, 4096, 'relu'))
        self.fc3 = nn.Sequential(*get_fc(4096, num_classes, 'none'))
 
    def forward(self, x):

        N = x.size(0)

        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.max_pool(x)    

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.nonlinear(x)

        x = self.conv4(x)
        x = self.nonlinear(x)

        x = self.conv5(x)
        x = self.nonlinear(x)
        x = self.max_pool(x)

        flat_x = x.view(N, 9216)
        flat_x = self.fc1(flat_x)
        flat_x = self.drop_out(flat_x)

        flat_x = self.fc2(flat_x)
        flat_x = self.drop_out(flat_x)

        out = self.fc3(flat_x)

        return out
    
def get_fc(inp_dim, out_dim, non_linear='relu'):

    layers = []
    layers.append(nn.Linear(inp_dim, out_dim))
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers

def main():
    a = torch.randn(1,3,227,227)
    model = CaffeNet()
    output = model(a)

if __name__ == "__main__":
    main()
        