import numpy as np 
import os

import torch
import torch.nn as nn
import torchvision.models as models

import utils
import imageio
from PIL import Image

from caffe_net import CaffeNet
from voc_dataset import VOCDataset

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

pool_features_list = []

def vis_image(dataset, index):
    findex = dataset.index_list[index]
    fpath = os.path.join(dataset.img_dir, findex + '.jpg')
    img = Image.open(fpath)
    img.show()

def feature_append(self, input, output):
    pool_feature_batch = output.data.cpu().numpy().reshape(output.shape[0],-1)
    pool_features_list.append(np.squeeze(pool_feature_batch)) 
    # pool_features.append(output.data) 

def nearest_neighbors():
    # PATH = "./checkpoints/resnet_pretrained.pth"
    # model = models.resnet18()
    # model.fc = nn.Linear(in_features=512, out_features=len(VOCDataset.CLASS_NAMES), bias=True)
    PATH = "./checkpoints/caffenet.pth"
    model = CaffeNet()
    model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=227, c_dim=3).to(device)
    

    model.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))
    model.eval()
    model = model.to(device)
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    model.max_pool5.register_forward_hook(feature_append)
    dataset = VOCDataset('test', 64)
    
    with torch.no_grad():
        for data, target, wgt in test_loader:   
            data, target, wgt = data.to(device), target.to(device), wgt.to(device)         
            output = model(data)

    pool_features = np.vstack(pool_features_list)

    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(pool_features)
    distances, indices = nbrs.kneighbors(pool_features)

    output_no = 0
    no_neighbors = 5
    selected_class_idx = []

    for i in range(pool_features.shape[0]):
        rand_idx = np.random.randint(low=0, high=pool_features.shape[0])
        class_label, _ = dataset.anno_list[rand_idx]
        class_label = np.squeeze(np.argwhere(class_label==1)).tolist()

        if len(list(set(selected_class_idx) & set(class_label))) == 0:
            selected_class_idx.extend(class_label)
            for i in range(no_neighbors):
                vis_image(dataset, indices[rand_idx,i])
        import pdb;pdb.set_trace()

def tsne():

    PATH = "./checkpoints/caffenet.pth"
    model = CaffeNet()
    model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=227, c_dim=3).to(device)
    

    model.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))
    model.eval()
    model = model.to(device)
    dataset = VOCDataset('test', 64)
    sample_size = 1000
    selected_idx = np.random.randint(low=0, high=dataset.__len__(), size=sample_size)
    test_loader = utils.get_data_loader('voc', train=False, batch_size=dataset.__len__(), split='test')

    model.fc2.register_forward_hook(feature_append)

    with torch.no_grad():
        for data, target, wgt in test_loader:   
            data, target, wgt = data.to(device), target.to(device), wgt.to(device)  
            data = data[selected_idx]  
            target = target[selected_idx]
            output = model(data)
    
    target_label = np.zeros(sample_size)
    
    for i in range(sample_size):
        target_label[i] = int(np.mean(np.argwhere(target[i].data.cpu().numpy())))
    features = np.vstack(pool_features_list)
    pca = PCA(n_components=100)
    reduced_features = pca.fit_transform(features)
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, n_jobs=8)
    tsne_results = tsne.fit_transform(reduced_features)

    df = pd.DataFrame(data=tsne_results, columns=['tsne-2d-one','tsne-2d-two'])

    df['classes'] = target_label
    plt.figure(figsize=(16,20))

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="classes",
        palette=sns.color_palette("hls", 20),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()


def main():
    # nearest_neighbors()
    tsne()
    
if __name__ == "__main__":
    args, device = utils.parse_args()
    main()