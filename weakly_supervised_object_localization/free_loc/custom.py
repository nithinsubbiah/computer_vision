import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

from PIL import Image
import os
import os.path
import numpy as np
import cPickle as pkl


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(imdb):
    #TODO: classes: list of classes
    #TODO: class_to_idx: dictionary with keys=classes and values=class index
    #If you did Task 0, you should know how to set these values from the imdb

    classes = list(imdb.classes)
    class_to_idx = imdb._class_to_ind

    return classes, class_to_idx


def make_dataset(imdb, class_to_idx):
    #TODO: return list of (image path, list(+ve class indices)) tuples
    #You will be using this in IMDBDataset

    dataset_list = []
    # class indices 1-20
    for idx in range(len(imdb._image_index)):
        annotation = imdb._load_pascal_annotation(imdb._image_index[idx])
        dataset_list.append((imdb.image_path_from_index(imdb._image_index[idx]),annotation['gt_classes'].to_list()))

    return dataset_list


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model

        self.features = nn.Sequential(
                            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3,stride=2,dilation=1,ceil_mode=False),
                            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3,stride=2,dilation=1,ceil_mode=False),
                            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            nn.ReLU(inplace=True)
                        )
        self.features.apply(self.init_weights)

        self.classifier = nn.Sequential(
                            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
                        )
        self.classifier.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_uniform(m.weight.data)
            # nn.init.xavier_uniform(m.bias.data)
 
    def forward(self, x):
        
        x = self.features(x)
        x = self.classifier(x)

        return x

class LocalizerAlexNetHighres(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetHighres, self).__init__()
        #TODO: Ignore for now until instructed


    def forward(self, x):
        #TODO: Ignore for now until instructed


        return x


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    #TODO: Initialize weights correctly based on whethet it is pretrained or
    # not
    ###TODO: Check
    if(pretrained):
        if os.path.exists('pretrained_alexnet.pkl'):
            model = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
        else:   
            model = model_zoo.load_url(
                'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
            pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
                    pkl.HIGHEST_PROTOCOL)
        
    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed

    return model
  

class IMDBDataset(data.Dataset):
    """A dataloader that reads imagesfrom imdbs
    Args:
        imdb (object): IMDB from fast-rcnn repository
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, list(+ve class indices)) tuples
    """

    def __init__(self, imdb, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(imdb)
        imgs = make_dataset(imdb, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images, what's going on?"))
        self.imdb = imdb
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a binary vector with 1s
                                   for +ve classes and 0s for -ve classes
                                   (it can be a numpy array)
        """
        # TODO: Write this function, look at the imagenet code for inspiration
        img_path, gt_classes = imgs[index]
        img = Image.open(img_path)
        img = self.transform(img)
        #TODO: target_transform??

        target = np.zeros(imdb.num_classes)
        for idx in gt_classes:
            target[idx-1] = 1

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
