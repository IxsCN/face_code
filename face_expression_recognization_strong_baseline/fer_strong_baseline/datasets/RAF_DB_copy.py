# encoding:utf-8

import numpy as np
import cv2
import pandas as pd
import pdb
import torch
from torch.utils.data import Dataset
from fer_strong_baseline.datasets.aug import fer_test_aug,fer_train_aug
from  fer_strong_baseline.config.default_cfg import  get_fer_cfg_defaults
from  fer_strong_baseline.utils.common import setup_seed
from tqdm import tqdm
import scipy.misc as sm
from torchvision import transforms
import os
import random
from fer_strong_baseline.datasets.aug import add_gaussian_noise, flip_image


class RafDataSet_white_gaussian(Dataset):
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        if self.is_train:
            self.data_lst = cfg.DATA.train_label_path
            self.aug = fer_train_aug(cfg.DATA.input_size,
                                     cfg.DATA.crop_residual_pix)
            self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((cfg.DATA.input_size, cfg.DATA.input_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                            transforms.RandomErasing(scale=(0.02, 0.25))
                            ])
        else:
            self.data_lst = cfg.DATA.val_label_path
            self.aug = fer_test_aug(cfg.DATA.input_size)
            self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((cfg.DATA.input_size, cfg.DATA.input_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])

        self.gaussian_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((cfg.DATA.input_size, cfg.DATA.input_size)),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[0],
                            #                      std=[255])
                            ])

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(self.data_lst, sep=' ', header=None)

        if self.is_train:
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
            # dataset = df[df[NAME_COLUMN].str.startswith('test')]

        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]

        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.labels = dataset.iloc[:,
                     LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        self.labels = np.array(self.labels)

        # use raf aligned images for training/testing
        self.samples = [os.path.join(cfg.DATA.img_dir,
                                     x.split('\n')[0].split(' ')[0].replace(".jpg", "_aligned.jpg"))
                        for x in file_names]

        self.rafdb_cls_num = ['surprised', 'fear', 'disgust', 'happy',
                   'sad', 'anger', 'normal']

        self.aug_func = [flip_image, add_gaussian_noise]

    def __len__(self):
        assert len(self.samples) == len(self.labels)
        return len(self.samples)


    def __getitem__(self, idx):

        path = self.samples[idx]
        image = cv2.imread(path)
        h, w, _ = image.shape
        # gui = image.copy()
        # print(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.labels[idx]
        # landmark_path = os.path.join(os.path.dirname(os.path.dirname(self.data_lst)), 'Annotation', 'manual', os.path.basename(path).replace('_aligned.jpg', '_manu_attri.txt'))

        # with open(landmark_path, 'r') as f:
        #     sample = f.readlines()
        # sample = [[ float(x) for x in p.split('\n')[0].replace(' ', '\t').split('\t')] for p in sample]
        # landmark = sample[:5]

        # gender = sample[5]
        # race = sample[6]
        # age = sample[7]

        # for p in landmark:
        #     cv2.circle(gui, (int(p[0]), int(p[1])), 3, (0,255,0), 3)
        # cv2.namedWindow('gui', 0)
        # cv2.imshow('gui', gui)
        # cv2.waitKey(0)
        # pdb.set_trace()
        # landmark =[[p[0]/h*224, p[1]/w*224] for p in landmark]
        gaussian_img = np.ones(image.shape, np.uint8) * 255

        # augmentation
        if self.is_train:
            if random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)
        # image = self.aug(image=image)['image']
        image = self.transform(image)
        gaussian_mask = self.gaussian_transform(gaussian_img)

        input_image = torch.cat((image, gaussian_mask), dim = 0)
        return input_image, label, idx
