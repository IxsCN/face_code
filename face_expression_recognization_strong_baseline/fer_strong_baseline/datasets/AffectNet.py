# encoding:utf-8

import numpy as np
import cv2
import pandas as pd
import pdb
import torch
from torch.utils.data import Dataset
from fer_strong_baseline.datasets.aug import fer_test_aug,fer_train_aug
from fer_strong_baseline.config.default_cfg import  get_fer_cfg_defaults
from fer_strong_baseline.utils.common import setup_seed
from fer_strong_baseline.datasets.aug import add_gaussian_noise, flip_image

from tqdm import tqdm
import scipy.misc as sm
import os
from torchvision import transforms
import random



class AffectNet_Dataset(Dataset):
    def __init__(self, cfg, is_train=True):
        self.affect_cls_num = ['normal', 'happy', 'sad', 'surprised',
                                 'fear', 'disgust', 'anger', 'contempt']
        
        self.rafdb_cls_num = ['surprised', 'fear', 'disgust', 'happy',
                       'sad', 'anger', 'normal']

        self.affect2rafdb = {0:6, 1:3, 2:4, 3:0, 4:1, 5:2, 6:5, 7:-1}
        
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

        # with open(self.data_lst, 'r') as f:
        #     samples = f.readlines()
        # self.samples = [x.split('\n')[0].split(',')[0] for x in samples]
        # self.labels = [x.split('\n')[0].split(',')[1] for x in samples]
        # self.labels = [str(self.affect2rafdb[int(x)]) for x in self.labels]
        # self.samples_and_labels = [[x,y] for (x,y) in zip(self.samples, self.labels) if int(y)>=0]
        # self.samples = [x for x,y in self.samples_and_labels]
        # self.labels = [y for x,y in self.samples_and_labels]

        data = pd.read_csv(self.data_lst)

        data = data[data['expression']<8]  #  # none, uncertain and non-face

        subDirectory_filePath = data['subDirectory_filePath']
        # face_x = data['face_x']
        # face_y = data['face_y']
        # face_width = data['face_width']
        # face_height = data['face_height']
        # facial_landmarks = data['facial_landmarks']
        expression = data['expression']
        # valence = data['valence']
        # arousal = data['arousal']

        error_img_lst = [
            "2/9db2af5a1da8bd77355e8c6a655da519a899ecc42641bf254107bfc0.jpg",
            "994/8fb7a123f36ae89be4bf72e9fa77d1434a29135f3342f62fb55c7aee.jpg",
        ]

        self.samples = subDirectory_filePath.tolist()
        self.labels = expression.tolist()

        self.samples_and_labels = [[os.path.join(cfg.DATA.img_dir, x),y] for (x,y) in zip(self.samples, self.labels) if x not in error_img_lst]

        self.samples = [x[0] for x in self.samples_and_labels]
        self.labels = [x[1] for x in self.samples_and_labels]


        all_data_len = len(subDirectory_filePath)

        self.labels = np.array(self.labels,  dtype=np.int64)

        self.aug_func = [flip_image, add_gaussian_noise]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_fn = self.samples[idx]
        # """
        # img_file = "/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated/Manually_Annotated_Images"
        # save_img_file = "/media/yz/62C9BA4E5D826344/data/AffectNet/Manually_Annotated/Face_Images"
        # """
        # img_fn.replace("Manually_Annotated_Images", "Face_Images")
        label = np.array(self.labels[idx].tolist(), dtype=np.int64)
        try:
            image = cv2.imread(img_fn)
            label = torch.from_numpy(label)
            assert len(image.shape) == 3
        except Exception as e:
            print("error img:{}".format(img_fn))
            pdb.set_trace()

        # augmentation
        if self.is_train:
            if random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)
        # image = self.aug(image=image)['image']
        image = self.transform(image)

        # label = label.float()
        # image = image.float()
        return image, label, idx, img_fn


class Make_Offline_AffectNet_Dataset(Dataset):
    def __init__(self, label_folder, img_folder, is_train=True):
        # ['subDirectory_filePath', 'face_x', 'face_y', 'face_width',
        #        'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal']
        if is_train:
            data_dir = os.path.join(label_folder, 'training.csv')
        else:
            data_dir = os.path.join(label_folder, 'validation.csv')

        data = pd.read_csv(data_dir)
        subDirectory_filePath = data['subDirectory_filePath']
        face_x = data['face_x']
        face_y = data['face_y']
        face_width = data['face_width']
        face_height = data['face_height']
        facial_landmarks = data['facial_landmarks']
        expression = data['expression']
        valence = data['valence']
        arousal = data['arousal']

        all_data_len = len(subDirectory_filePath)

        if is_train:
            data_lst = os.path.join(label_folder, "train.txt")
        else:
            data_lst = os.path.join(label_folder, "val.txt")

        for i in tqdm(range(all_data_len)):
            if expression[i] > 7:  # none, uncertain and non-face
                continue

            img_fn = os.path.join(img_folder, subDirectory_filePath[i])
            save_fn = img_fn.replace('Manually_Annotated_Images', 'Manually_Annotated_Face_Images')
            if not os.path.exists(os.path.dirname(save_fn)):
                os.makedirs(os.path.dirname(save_fn))
            else:
                if os.path.exists(save_fn):
                    f = open(data_lst, 'ImbalancedDatasetSampling+')
                    f.write(save_fn + ',' + str(expression[i]) + '\n')
                    f.close()
                    continue


            try:
                x0, y0, w, h = int(face_x[i]), int(face_y[i]), int(face_width[i]), int(face_height[i])
                src = cv2.imread(img_fn)
                face_img = src[x0:x0+w-1, y0:y0+h-1, :]
                # pdb.set_trace()
                cv2.imwrite(save_fn, face_img)
                f = open(data_lst, 'ImbalancedDatasetSampling+')
                f.write(save_fn + ',' + str(expression[i]) + '\n')
                f.close()
            except Exception as e:
                continue

        # [{'neutral': (-0.019574175715601782, -0.0627188977615683)},
        # {'happy': (0.07028056723210271, 0.6643268644924916)},
        # {'sad': (-0.2570043582489509, -0.6370063573392657)},
        # {'surprise': (0.6893279744258441, 0.1804439995542943)},
        # {'fear': (0.7661802158670421, -0.12574468561147706)},
        # {'disgust': (0.4573438234998662, -0.693718996983959)},
        # {'anger': (0.5667776418527468, -0.45273569245960493)},
        # {'contempt': (0.5828403601946657, -0.5145737456826646)},
        # {'None': (-0.16301792557996833, 0.1225122491637435)},
        # {'Uncertain': (-2.0, -2.0)},
        # {'Non-Face': (-2.0, -2.0)}]

if __name__ == "__main__":
    setup_seed(666)
    cfg = get_fer_cfg_defaults()
    config_path = '/configs/affectnet_res50_ce.yml'
    cfg.merge_from_file(config_path)

    label_path = '/media/seven/7EF450DF596F8A46/data/AffectNet/Manually_Annotated_file_lists'
    img_path = '/media/seven/7EF450DF596F8A46/data/AffectNet/Manually_Annotated/Manually_Annotated_Images'
    # Make_Offline_AffectNet_Dataset(label_path, img_path, False)

    dataset = AffectNet_Dataset(cfg, True)
    for i in dataset:
        pdb.set_trace()
