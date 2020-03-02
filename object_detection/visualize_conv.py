import numpy as np
import glob

import torch
import matplotlib.pyplot as plt

from caffe_net import CaffeNet
from voc_dataset import VOCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "./checkpoints/model*.pth"
model_list = glob.glob(PATH)
model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=227, c_dim=3).to(device)

for model_path in model_list: 
	model_name = (model_path.split('/')[-1]).split('.')[0]
	model.load_state_dict(torch.load(model_path))

	# Print model's state_dict
	# print("Model's state_dict:")
	# for param_tensor in model.state_dict():
	#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

	weight_tensor = model.conv1.weight.data
	print(weight_tensor.shape)

	no_filters = 5
	num_rows = 1
	num_cols = no_filters

	fig = plt.figure(figsize=(num_cols,num_rows))

	# code apapted from: https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
	for i in range(no_filters):
    		ax1 = fig.add_subplot(num_rows,num_cols,i+1)
    		#for each kernel, we convert the tensor to numpy 
    		npimg = np.array(weight_tensor[i].cpu().numpy(), np.float32)
    		#standardize the numpy image
    		npimg = (npimg - np.mean(npimg)) / np.std(npimg)
    		npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
    		npimg = npimg.transpose((1, 2, 0))
    		ax1.imshow(npimg)
    		ax1.axis('off')
    		ax1.set_title(str(i))
    		ax1.set_xticklabels([])
    		ax1.set_yticklabels([])

	plt.savefig('conv1'+str(model_name)+'.png', dpi=100)    
	plt.tight_layout()
	plt.show()

