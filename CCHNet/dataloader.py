from torchvision import transforms
import torch
import pandas as pd
import PIL
from PIL import Image
import torch.utils.data as data
from torch.utils.data import Dataset
# from torchtoolbox.transform import Cutout
# from Guide_feature import GuidedFilter
import random
from pathlib import Path
# import cv2
from utils.variables import *
from my_transforms import AddGaussianNoise, AddSaltPepperNoise
import numpy as np



def bulid_neg_transform(type, args):
    if type == "train":
        compose = transforms.Compose([
            transforms.Resize((args.image_size + args.padding_size, args.image_size + args.padding_size)),
            transforms.RandomCrop(args.crop_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            AddGaussianNoise(mean=0.0, variance=1.0, amplitude=1.0, prob=0.2),
            AddSaltPepperNoise(density=0.2, prob=0.2),
            # transforms.ColorJitter(saturation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    else:
        compose = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return compose

def bulid_pos_transform(type, args):
    if type == "train":
        compose = transforms.Compose([
            transforms.Resize((args.image_size + args.padding_size, args.image_size + args.padding_size)),
            transforms.RandomCrop(args.crop_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            # AddGaussianNoise(mean=0.0, variance=1.0, amplitude=1.0, prob=0.2),
            # AddSaltPepperNoise(density=0.2, prob=0.2),
            # transforms.ColorJitter(saturation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    else:
        compose = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return compose


class CCHSeT_CSV(Dataset):

    def __init__(self, root, type_, args):
        self.root = Path(root)
        self.type_ = type_
        if type_ == "train":
            self.csv = self.root / "cch_train_4.csv"
            self.transform_pos = bulid_pos_transform(type_, args)
            self.transform_neg = bulid_neg_transform(type_, args)
            # self.transform_pos = train_compose_pos
        elif type_ == "test":
            self.csv = self.root / "cch_test_4.csv"
            self.transform_pos = bulid_pos_transform(type_, args)
            self.transform_neg = bulid_neg_transform(type_, args)
            # self.transform_neg = test_compose
        elif type_ == "val":
            self.csv = self.root / "cch_val.csv"
            self.transform_pos = bulid_pos_transform(type_, args)
            self.transform_neg = bulid_neg_transform(type_, args)
            # self.transform_neg = test_compose
        self.check_files(self.csv)
        try:
            self.csv = pd.read_csv(self.csv)
        except:
            self.csv = pd.read_csv(self.csv, encoding='gbk')
        self.csv = self.csv.dropna()
        self.csv['name'] = self.csv['name'].astype(str)
        self.people_classfiy = self.csv.loc[:, 'cch'].map(lambda x: 1 if (x // 1) >= 1 else 0)
        self.people_classfiy.index = self.csv['name']
        self.people_classfiy = self.people_classfiy.to_dict()

        self.neg_pic = []
        self.pos_pic = []
        self.pic_files = []
        for p in self.people_classfiy:
            if type_ == 'train':
                pic_file = self.root / str(p)
                pic_file_t1 = pic_file / 't1'
                pic_file_t2 = pic_file / 't2'
                pic_file = list(pic_file_t1.rglob('*.bmp')) + list(pic_file_t2.rglob('*.bmp'))
            else:
                pic_file = self.root / str(p)
                pic_file_t1 = pic_file / 't1'
                pic_file_t2 = pic_file / 't2'
                pic_file = list(pic_file_t1.rglob('*.bmp')) + list(pic_file_t2.rglob('*.bmp'))
            self.pic_files += pic_file

            if self.people_classfiy[p] == 1:
                self.pos_pic += pic_file
            else:
                self.neg_pic += pic_file


        self.filter32()

        if type_ == 'train':
            print(len(self.neg_pic), len(self.pos_pic), len(self.pic_files))
            if len(self.pos_pic) >= len(self.neg_pic):
                ratio = int(len(self.pos_pic) // len(self.neg_pic))
                distance = len(self.pos_pic) - (ratio * len(self.neg_pic))
                random.shuffle(self.neg_pic)
                self.neg_pic = (ratio) * self.neg_pic + self.neg_pic[0: distance]
                # self.pos_pic = (ratio + 1) * self.pos_pic
                # self.pos_pic = self.pos_pic[0: len(self.neg_pic)]
                self.pic_files = self.pos_pic + self.neg_pic
                random.shuffle(self.pic_files)
            else:
                ratio = int(len(self.neg_pic) // len(self.pos_pic))
                distance = len(self.neg_pic) - (ratio * len(self.pos_pic))
                random.shuffle(self.pos_pic)
                self.pos_pic = (ratio) * self.pos_pic + self.pos_pic[0: distance]
                self.pic_files = self.pos_pic + self.neg_pic
                random.shuffle(self.pic_files)
        else:
            self.pic_files = self.pos_pic + self.neg_pic
        self.pic_files = self.pos_pic + self.neg_pic
        print(len(self.neg_pic), len(self.pos_pic))

    def check_files(self, file):
        assert Path(file).exists(), FileExistsError('{str(file)}不存在')

    def filter32(self):
        neg_files = []
        pos_files = []
        for pic in self.neg_pic:
            img = Image.open(pic)
            if img.width >= 32 and img.height >= 32:
                neg_files.append(pic)
        self.neg_pic = neg_files
        for pic in self.pos_pic:
            img = Image.open(pic)
            if img.width >= 32 and img.height >= 32:
                pos_files.append(pic)
        self.pos_pic = pos_files

    def __len__(self):
        return len(self.pic_files)

    def __getitem__(self, index):
        img_single = Image.open(str(self.pic_files[index])).convert("RGB")
        people = str(self.pic_files[index].parent.parent.name)
        cch = self.csv.loc[self.csv['name'] == str(people), "cch"].iloc[0]
        name = str(people)
        y = self.people_classfiy[str(people)]
        # img_data = self.transform(img_single)
        if y == 0:
            img_data = self.transform_neg(img_single)
        else:
            img_data = self.transform_pos(img_single)
        rs = {
            "img": img_data,
            "label": torch.Tensor([y])[0],
            "name": name,
            "cch": cch,
            "image_path": str(self.pic_files[index])
        }
        return rs