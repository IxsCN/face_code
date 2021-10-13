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



class RafDataSet_v2(Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(self.raf_path, sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.labels = dataset.iloc[:,
                     LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(os.path.dirname(os.path.dirname(self.raf_path)), 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        # print(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.labels[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx

class RafDataSet(Dataset):
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        if self.is_train:
            self.data_lst = cfg.DATA.train_label_path
            self.aug = fer_train_aug(cfg.DATA.input_size,
                                     cfg.DATA.crop_residual_pix)
            if cfg.DATA.input_channel == 3:
                self.transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((cfg.DATA.input_size, cfg.DATA.input_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                transforms.RandomErasing(scale=(0.02, 0.25))
                                ])
            else:  # gray
                self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Grayscale(1),
                            transforms.Resize((cfg.DATA.input_size + 10, cfg.DATA.input_size + 10)),
                            transforms.RandomCrop(cfg.DATA.input_size),
                            transforms.RandomRotation(10),
                            transforms.RandomHorizontalFlip(p=0.5),  
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #                      std=[0.229, 0.224, 0.225]),
                            # transforms.RandomErasing(scale=(0.02, 0.25))
                            ])
        else:
            self.data_lst = cfg.DATA.val_label_path
            self.aug = fer_test_aug(cfg.DATA.input_size)

            if cfg.DATA.input_channel == 3:
                self.transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((cfg.DATA.input_size, cfg.DATA.input_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
            else:
                self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Grayscale(1),
                            transforms.Resize((cfg.DATA.input_size + 10, cfg.DATA.input_size + 10)),
                            transforms.TenCrop(cfg.DATA.input_size),  
                            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),    
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #                      std=[0.229, 0.224, 0.225]),
                            # transforms.RandomErasing(scale=(0.02, 0.25))
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
        # use raf aligned images for training/testing
        self.aligned_samples = [os.path.join(cfg.DATA.img_dir,
                                     x.split('\n')[0].split(' ')[0].replace(".jpg", "_aligned.jpg"))
                        for x in file_names]
        self.samples = self.aligned_samples
        # self.face_crop_samples = [os.path.join(cfg.DATA.img_dir.replace("aligned", "face_crop"), 
        #                              x.split('\n')[0].split(' ')[0])
        #                 for x in file_names]
        # # 有些图没有检测到landmark，此时只能用aligned image替代
        # self.samples_and_labels = [{x1:y}  if os.path.exists(x1) else {x2:y} for (x1, x2, y) in zip(self.face_crop_samples, self.aligned_samples, self.labels)]

        # self.samples = [list(x.keys())[0] for x in self.samples_and_labels]
        # self.labels = [list(x.values())[0] for x in self.samples_and_labels]

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
        landmark_path = os.path.join(os.path.dirname(os.path.dirname(self.data_lst)), 'Annotation', 'manual', os.path.basename(path).replace('_aligned.jpg', '_manu_attri.txt'))

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

        # augmentation
        if self.is_train:
            if random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)
        # image = self.aug(image=image)['image']
        try:
            image = self.transform(image)
        except:
            import pdb; pdb.set_trace()

        return image, label, idx, path
