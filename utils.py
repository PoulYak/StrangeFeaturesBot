import torchvision.transforms as transforms
from PIL import Image
import os
from torch.nn import init
import numpy as np
import copy


class Dataset():
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.filenames = sorted([i for i in os.listdir(self.directory)])
        self.len_ = len(self.filenames)
        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((286, 286)),
             transforms.RandomCrop((256, 256)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
             ])

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.directory, self.filenames[index])).convert('RGB')
        x = self.transform(x)
        return x

    def __len__(self):
        return self.len_


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('Network was initialized')
    net.apply(init_func)


class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


class LambdaLR():
    def __init__(self, ep, off, dec_ep):
        self.ep = ep
        self.off = off
        self.dec_ep = dec_ep

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.off - self.dec_ep) / (self.ep - self.dec_ep)
