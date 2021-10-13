import numpy as np
import cv2
import pandas as pd
import os
import pdb
import torch
from torch.utils.data import  Dataset
from fer_strong_baseline.datasets.aug import fer_test_aug,fer_train_aug
from  fer_strong_baseline.config.default_cfg import  get_fer_cfg_defaults
from  fer_strong_baseline.utils.common import setup_seed

# encoding:utf-8
import pandas as pd
import numpy as np
import scipy.misc as sm
import os

emotions = {
    '0': 'anger',  # 生气
    '1': 'disgust',  # 厌恶
    '2': 'fear',  # 恐惧
    '3': 'happy',  # 开心
    '4': 'sad',  # 伤心
    '5': 'surprised',  # 惊讶
    '6': 'normal',  # 中性
}


# 创建文件夹
def createDir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)


def saveImageFromFer2013(file):
    # 读取csv文件
    faces_data = pd.read_csv(file)
    imageCount = 0
    # 遍历csv文件内容，并将图片数据按分类保存
    for index in range(len(faces_data)):
        # 解析每一行csv文件内容
        emotion_data = faces_data.loc[index][0]
        image_data = faces_data.loc[index][1]
        usage_data = faces_data.loc[index][2]
        # 将图片数据转换成48*48
        data_array = list(map(float, image_data.split()))
        data_array = np.asarray(data_array)
        image = data_array.reshape(48, 48)

        # 选择分类，并创建文件名
        dirName = usage_data
        emotionName = emotions[str(emotion_data)]

        # 图片要保存的文件夹
        imagePath = os.path.join(dirName, emotionName)

        # 创建“用途文件夹”和“表情”文件夹
        createDir(dirName)
        createDir(imagePath)

        # 图片文件名
        imageName = os.path.join(imagePath, str(index) + '.jpg')
        pdb.set_trace()
        exit(0)
        # sm.toimage(image).save(imageName)
        cv2.imwrite(imageName, image)
        imageCount = index
    print('总共有' + str(imageCount) + '张图片')

def get_img_from_folder(img_folder):
    for _root, _dir, _file in os.walk(img_folder):
        pdb.set_trace()
        if not os.path.isdir(_file):
            pass

class FER2013FolderDataset(Dataset):
    # 28709 picture
    h, w = 48, 48  # h must equal as w
    num_cls = 7
    x_train, y_train, x_test, y_test = [], [], [], []

    def __init__(self, cfg, is_train=True):
        data = get_img_from_folder(cfg.DATA.train_label_path)  # only one file

        data = pd.read_csv(cfg.DATA.train_label_path)  # only one file
        self.pixels = data['pixels']  # 48*48
        self.emotions = data['emotion']  # 0~6
        self.usages = data['Usage']  # 0,1
        emo_tmp = [0 for _ in range(self.num_cls)]
        for emo, img, usage in zip(self.emotions, self.pixels, self.usages):
            try:
                # emo_tmp1 = emo_tmp.copy()
                # emo_tmp1[emo] = 1
                emo_tmp1 = [emo]
                img = [0 for _ in img.split(" ")]
                if usage == 'Training':
                    self.x_train.append(img)
                    self.y_train.append(emo_tmp1)

                else:
                    self.x_test.append(img)
                    self.y_test.append(emo_tmp1)

            except Exception as e:
                pdb.set_trace()
        pdb.set_trace()
        self.x_train = np.array(self.x_train, dtype=np.uint8).reshape(-1, self.h, self.w)
        self.x_test = np.array(self.x_test, dtype=np.uint8).reshape(-1, self.h, self.w)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        # pdb.set_trace()
        if is_train == True:
            # self._len = self.x_train.shape[0]
            # self.x = self.x_train
            # self.y = self.y_train
            self._len = self.x_test.shape[0]
            self.x = self.x_test
            self.y = self.y_test
            self.aug = fer_train_aug(cfg.DATA.input_size,
                                     cfg.DATA.crop_residual_pix)
        else:
            self._len = self.x_test.shape[0]
            self.x = self.x_test
            self.y = self.y_test

            self.aug = fer_test_aug(cfg.DATA.input_size)

    def __len__(self):
        return self._len

class FER2013Dataset(Dataset):
    # 28709 picture
    h, w = 48, 48  # h must equal as w
    num_cls = 7
    # 0 anger 生气； 1 disgust 厌恶； 2 fear 恐惧； 3 happy 开心； 4 sad 伤心；5 surprised 惊讶； 6 normal 中性
    x_train, y_train, x_test, y_test = [], [], [], []
    def __init__(self, cfg, is_train=True):
        data = pd.read_csv(cfg.DATA.train_label_path)  # only one file
        self.pixels = data['pixels']  # 48*48
        self.emotions = data['emotion']  # 0~6
        self.usages = data['Usage']  # 0,1
        emo_tmp = [0 for _ in range(self.num_cls)]
        train_cls_cnt = [0 for _ in range(self.num_cls)]
        val_cls_cnt = [0 for _ in range(self.num_cls)]
        for emo, img, usage in zip(self.emotions, self.pixels, self.usages):
            try:

                # emo_tmp1 = emo_tmp.copy()
                # emo_tmp1[emo] = 1
                emo_tmp1 = [emo]
                img = [x for x in img.split(" ")]
                if usage == 'Training':
                    train_cls_cnt[emo] += 1

                    self.x_train.append(img)
                    self.y_train.append(emo_tmp1)
                elif usage == 'PublicTest':
                    val_cls_cnt[emo] += 1

                    self.x_test.append(img)
                    self.y_test.append(emo_tmp1)
                else:
                    pass

            except Exception as e:
                pdb.set_trace()
        # pdb.set_trace()
        print("training data:", train_cls_cnt)
        print("val data:", val_cls_cnt)

        self.x_train = np.array(self.x_train, dtype=np.uint8).reshape(-1, self.h, self.w)
        self.x_test = np.array(self.x_test, dtype=np.uint8).reshape(-1,  self.h, self.w)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        # pdb.set_trace()
        if is_train == True:
            self._len = self.x_train.shape[0]
            self.x = self.x_train
            self.y = self.y_train
            # self._len = self.x_test.shape[0]
            # self.x = self.x_test
            # self.y = self.y_test
            self.aug = fer_train_aug(cfg.DATA.input_size,
                                     cfg.DATA.crop_residual_pix)
        else:
            self._len = self.x_test.shape[0]
            self.x = self.x_test
            self.y = self.y_test

            self.aug = fer_test_aug(cfg.DATA.input_size)
    def __len__(self):
        return self._len

    def __getitem__(self, item):
        img = self.x[item, :, :]
        label = self.y[item, :]
        try:
            # img = img.astype()
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = self.aug(image=img)['image']
            img = img.transpose((2, 0, 1))  # nchw
            img = torch.from_numpy(img)
            label = torch.from_numpy(label)
        except Exception as e:
            pdb.set_trace()
        return img, label



if __name__ == "__main__":
    data_path = "/home/seven/data/fer_data/fer2013/fer2013.csv"
    # saveImageFromFer2013(data_path)

    setup_seed(666)
    cfg = get_fer_cfg_defaults()
    config_path = '/configs/fer2013_res50_ce.yml'
    cfg.merge_from_file(config_path)

    fer_dataset = FER2013Dataset(cfg, True)
    for i in fer_dataset:
        pdb.set_trace()

