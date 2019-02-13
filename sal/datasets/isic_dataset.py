import os
import pickle
import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose, CenterCrop, ToTensor, RandomHorizontalFlip, RandomSizedCrop,
    ColorJitter, RandomVerticalFlip
)


ISIC_EVAL_TRANSFORMS = Compose([
                                CenterCrop(224),
                                ToTensor()]
                               )
ISIC_IMAGES_DIR = '/home/xinyang/Data/intattack/ISIC-2018/data/isic/task3/images/HAM10000/'
ISIC_LABEL_PATH = '/home/xinyang/Data/intattack/ISIC-2018/data/isic/task3/labels/HAM10000/labels.csv'
ISIC_CV_SPLIT_PATH = '/home/xinyang/Data/intattack/ISIC-2018/data/isic/indices_new.pkl'

ISIC_RESNET50_CKPT_PATH = '/home/xinyang/Data/intattack/ISIC-2018/data/isic/ISIC.example_resnet50_5foldcv/CVSet1/checkpoint_best-65.pt'


def read_metadata(path):
    reader = csv.reader(open(path))
    it = iter(reader)
    next(it)
    images, labels = [], []
    for row in it:
        image, label = row[0], [int(float(v)) for v in row[1:]]
        label = label.index(1)
        images.append(image + '.jpg')
        labels.append(label)
    return images, np.asarray(labels, np.int64)


def get_isic_train_transform():
    transforms = Compose([
                        RandomSizedCrop(224),
                        RandomHorizontalFlip(),
                        RandomVerticalFlip(),
                        ColorJitter(brightness=32. / 255.,saturation=0.5),
                        ToTensor()
    ])
    return transforms


def get_isic_val_transform():
    transforms = Compose([
                         CenterCrop(224),
                         ToTensor()
    ])
    return transforms


def get_train_dataset(size=224):
    with open(ISIC_CV_SPLIT_PATH, 'rb') as f:
        obj = pickle.load(f)
    indices = obj['trainIndCV'][1]
    return ISICDataset(ISIC_IMAGES_DIR, None, indices, transforms=get_isic_train_transform())


def get_val_dataset(size=224):
    with open(ISIC_CV_SPLIT_PATH, 'rb') as f:
        obj = pickle.load(f)
    indices = obj['valIndCV'][1]
    return ISICDataset(ISIC_IMAGES_DIR, None, indices, transforms=get_isic_val_transform())


def get_loader(dataset, batch_size=64, pin_memory=True):
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=True, drop_last=True, num_workers=4, pin_memory=pin_memory)


class ISICDataset(Dataset):

    def __init__(self, images_dir=None, label_path=None, indices=None, transforms=None):
        if images_dir is None:
            images_dir = ISIC_IMAGES_DIR
        self.image_dir = images_dir
        if label_path is None:
            label_path = ISIC_LABEL_PATH
        self.image_paths, self.labels = read_metadata(label_path)
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = self.labels[indices]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(path)
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[index]
        return image, label
