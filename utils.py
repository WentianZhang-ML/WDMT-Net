import json
import math
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image, ImageStat
from torchvision import transforms as T
import torchvision.transforms.functional as F
import os
import logging
import random


def set_log(path, time_,args):
    if not os.path.exists(path):
        os.makedirs(path)
    log_path = path + 'Time_' + str(time_) + 'lr_' + str(args.lr) + '_log.txt'
    logger = logging.getLogger("mainModule")
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(filename=log_path, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
def logg(args, time):
    log_path = args.savedir + args.dataset + '/' +args.model + '/training_logs/'
    logger = set_log(log_path, time, args)
    logger.info("Start Time:{}".format(time))
    logger.info("Model:{}".format(args.model))
    logger.info("Log Path:{}".format(log_path))
    logger.info("Model Savedir:{}".format(
        args.savedir + args.dataset + '/' +args.model + '/saved_models/Time_' + str(time) + '_lr_' + str(args.lr)+ '/'))
    logger.info("Training Protocol")
    logger.info("Epoch Total number:{}".format(args.epochs))
    logger.info("Batch Size is {:^.2f}".format(args.batch_size))
    logger.info("Learning Rate:{}".format(args.lr))
    return logger
def clip(image, mask):
    if random.random() > 0.5:
        image = F.hflip(image)
        mask = F.hflip(mask)
    if random.random() > 0.5:
        image = F.vflip(image)
        mask = F.vflip(mask)
    return image, mask

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

class Datasets_Train(Dataset):
    def __init__(self, data):
        self.data = data
        self.transforms = T.Compose([
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item]['image_path']
        image_hog_path = self.data[item]['image_hog_path']
        img_label = self.data[item]['image_normal_or_abormal_label']
        img_id = self.data[item]['image_ID']
        image = Image.open(image_path).convert('L')
        image_hog = Image.open(image_hog_path).convert('L')
        image, image_hog=clip(image, image_hog)
        image = self.transforms(image)
        image_hog = T.ToTensor()(image_hog)

        return image, img_id, image_hog, img_label
class Datasets_Test(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item]['image_path']
        img_id = self.data[item]['image_ID']
        image_hog_path = self.data[item]['image_hog_path']

        img_label = self.data[item]['image_ROI']
        image = Image.open(image_path).convert('L')
        image = T.ToTensor()(image)
        image_hog = Image.open(image_hog_path)
        image_hog = T.ToTensor()(image_hog)
        image_label = Image.open(img_label)
        image_label = T.ToTensor()(image_label)
        
        if 'ADAM' in image_path:
            # # ADAM
            image_label = torch.where(image_label >0, 0, 1)
        else:
            # # IDRiD
            image_label = torch.where(image_label >0, 1, 0)
        return image, img_id, image_hog, image_label



def load_data(dataset_name):
    train_path = json.load(open(dataset_name + '/train_label.json', 'r'))
    test_path = json.load(open(dataset_name + '/test_label.json', 'r'))
    return train_path, test_path

def get_dataset(dataset_name, batchsize):
    train_data, test_data = load_data(dataset_name)

    data_loader_train = DataLoader(Datasets_Train(train_data), batch_size=batchsize, shuffle=True, num_workers=32,
                                   pin_memory=True, drop_last=True)
    data_loader_test = DataLoader(Datasets_Test(test_data), batch_size=batchsize, shuffle=False, num_workers=32,
                                  pin_memory=True, drop_last=False)
    return data_loader_train, data_loader_test







